"""Pinned local encoders and guarded hosted embedding adapters for comparisons."""

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .rag import SentenceEmbedder


def document_text(item):
    title = item.get("title", "")
    return f"{title}\n{item['text']}" if title else item["text"]


def formatted_inputs(key, items, role, domain):
    if role not in {"query", "document"}:
        raise ValueError("embedding role must be query or document")
    content = [
        item["query"] if role == "query" else document_text(item) for item in items
    ]
    if key == "qwen" and role == "query":
        instruction = (
            "Given a scientific claim, retrieve scientific abstracts that provide evidence to verify the claim."
            if domain == "scifact"
            else "Given a question, retrieve relevant passages that answer the question."
        )
        return [f"Instruct: {instruction}\nQuery: {text}" for text in content]
    if key in {"embeddinggemma", "gemini"}:
        if role == "document":
            return [
                f"title: {item.get('title') or 'none'} | text: {item['text']}"
                for item in items
            ]
        task = "fact checking" if domain == "scifact" else "question answering"
        return [f"task: {task} | query: {text}" for text in content]
    return content


def checked_vectors(values, count, dimensions):
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (count, dimensions) or not np.isfinite(matrix).all():
        raise ValueError("embedding response has invalid shape or values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding response contains a zero vector")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def ordered_voyage_vectors(payload, count):
    rows = payload.get("data", [])
    indices = [r.get("index") for r in rows]
    if len(rows) != count or set(indices) != set(range(count)):
        raise ValueError("hosted response indices are incomplete or duplicated")
    return [r["embedding"] for r in sorted(rows, key=lambda row: row["index"])]


def checked_snapshot(spec):
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            spec["id"],
            revision=spec["revision"],
            allow_patterns=[
                "*.json",
                "*.py",
                "*.txt",
                "*.model",
                "*.jinja",
                "*.safetensors",
            ],
        )
    )
    for filename, expected in spec.get("reviewed_code", {}).items():
        if hashlib.sha256((path / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("custom model code differs from the inspected revision")
    return path


class StudyEncoder:
    def __init__(self, key, spec, *, api_budget_usd=0):
        self.key = key
        self.spec = spec
        self.api_budget_usd = api_budget_usd
        self.reserved_cost_usd = 0.0
        self.calls = []
        self.model = None
        self.tokenizer = None
        if spec["backend"] == "local":
            self._load_local()

    def _load_local(self):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(4)
        if self.key == "minilm":
            self.model = SentenceEmbedder(self.spec["id"], self.spec["revision"]).model
        else:
            path = checked_snapshot(self.spec)
            dtype = getattr(torch, self.spec["dtype"])
            if self.key == "jina":
                from transformers import AutoModel

                self.model = AutoModel.from_pretrained(
                    str(path), trust_remote_code=True, dtype=dtype
                )
                self.tokenizer = self.model.tokenizer
                self.tokenizer.padding_side = "right"
            else:
                config = (
                    {"vision_config": None, "audio_config": None}
                    if self.key == "embeddinggemma"
                    else {"use_cache": False}
                )
                self.model = SentenceTransformer(
                    str(path),
                    device="cpu",
                    model_kwargs={"dtype": dtype},
                    config_kwargs=config,
                )
        self.model.eval()
        if self.tokenizer is None:
            self.tokenizer = self.model.tokenizer

    def encode(self, items, role, domain):
        texts = formatted_inputs(self.key, items, role, domain)
        if self.spec["backend"] != "local":
            return self._encode_hosted(texts, role)
        # The native Jina encode method applies its Query/Document prefixes.
        actual = (
            [("Query: " if role == "query" else "Document: ") + text for text in texts]
            if self.key == "jina"
            else texts
        )
        lengths = [
            len(tokens)
            for tokens in self.tokenizer(
                actual, add_special_tokens=True, truncation=False
            )["input_ids"]
        ]
        if max(lengths, default=0) > self.spec["max_tokens"]:
            raise ValueError("formatted input exceeds the encoder limit")
        if self.key == "jina":
            values = self.model.encode(texts=texts, task="retrieval", prompt_name=role)
            values = values.detach().float().cpu().numpy()
        else:
            values = self.model.encode(
                texts,
                batch_size=self.spec["batch_size"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                prompt="",
            )
        return checked_vectors(values, len(items), self.spec["dimensions"])

    def _encode_hosted(self, texts, role):
        token = os.environ.get(self.spec["environment"])
        if not token:
            raise ValueError("hosted credentials are unavailable")
        if self.api_budget_usd <= 0:
            raise ValueError(
                "hosted execution requires an explicitly approved spending cap"
            )
        if self.spec["backend"] == "voyage":
            payload = {
                "model": self.spec["id"],
                "input": texts,
                "input_type": role,
                "truncation": False,
            }
            result = self._request(
                "https://api.voyageai.com/v1/embeddings",
                payload,
                {"Authorization": f"Bearer {token}"},
                texts,
                0.12,
            )
            values = ordered_voyage_vectors(result, len(texts))
        else:
            # Separate requests prevent the multimodal API from aggregating inputs.
            values = []
            for text in texts:
                payload = {
                    "model": f"models/{self.spec['id']}",
                    "content": {"parts": [{"text": text}]},
                }
                result = self._request(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.spec['id']}:embedContent",
                    payload,
                    {"x-goog-api-key": token},
                    [text],
                    0.20,
                )
                values.append(result["embedding"]["values"])
        return checked_vectors(values, len(texts), self.spec["dimensions"])

    def _request(self, url, payload, headers, texts, price):
        reservation = (
            sum(len(t.encode("utf-8")) + 128 for t in texts) * price / 1_000_000
        )
        for attempt in range(2):
            if self.reserved_cost_usd + reservation > self.api_budget_usd:
                raise ValueError(
                    "embedding request would exceed the approved conservative budget"
                )
            self.reserved_cost_usd += reservation
            start = time.perf_counter()
            status = "failed"
            try:
                request = Request(
                    url,
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", **headers},
                )
                with urlopen(request, timeout=120) as response:
                    result = json.load(response)
                status = "ok"
                return result
            except (HTTPError, URLError, TimeoutError, OSError):
                if attempt == 1:
                    raise RuntimeError(
                        "hosted embedding request failed after bounded retries"
                    ) from None
            finally:
                self.calls.append(
                    {
                        "attempt": attempt + 1,
                        "status": status,
                        "seconds": time.perf_counter() - start,
                        "reserved_usd": reservation,
                    }
                )
