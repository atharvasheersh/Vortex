"""Serve the local Vortex workbench and a real, read-only native search API."""

import argparse
import hashlib
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
from vortex import Index, __version__, avx2_available
from vortex.rag import FixtureEmbedder

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "README.md",
    "docs/architecture.md",
    "docs/file_format.md",
    "docs/evaluation.md",
    "docs/scalability.md",
    "docs/capacity_10m.md",
    "docs/real_rag.md",
    "docs/verification.md",
    "docs/workbench.md",
]
STATIC = {
    "/": ("index.html", "text/html"),
    "/styles.css": ("styles.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}


def read_documents(root, sources=SOURCES):
    documents = []
    for source in sources:
        path = root / source
        if not path.is_file():
            continue
        title = path.stem.replace("_", " ").capitalize()
        paragraph, start, fenced = [], 1, False

        def flush(paragraph, source, start, title):
            if not paragraph:
                return
            text = " ".join(paragraph).strip()
            if len(text) < 55:
                return
            identifier = int.from_bytes(
                hashlib.blake2b(f"{source}:{start}".encode(), digest_size=8).digest(),
                "little",
            )
            documents.append(
                {
                    "id": str(identifier),
                    "title": title,
                    "source": source,
                    "line": start,
                    "text": text,
                    "words": len(text.split()),
                }
            )

        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            if fenced:
                continue
            if line.startswith("#"):
                flush(paragraph, source, start, title)
                paragraph = []
                title = line.lstrip("# ").strip()
                continue
            if not line.strip():
                flush(paragraph, source, start, title)
                paragraph = []
                continue
            if sum(map(len, paragraph)) > 1100:
                flush(paragraph, source, start, title)
                paragraph = []
            if not paragraph:
                start = line_number
            plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line.strip())
            plain = re.sub(r"[*`#]", "", plain)
            paragraph.append(plain)
        flush(paragraph, source, start, title)
    if len(documents) < 2:
        raise ValueError("At least two documentation chunks are required")
    if len({d["id"] for d in documents}) != len(documents):
        raise ValueError("Documentation IDs must be unique")
    return documents


class Workbench:
    def __init__(self, root=ROOT, sources=SOURCES):
        self.root = Path(root)
        self.sources = list(sources)
        self.documents = read_documents(self.root, sources)
        self.lookup = {int(d["id"]): d for d in self.documents}
        self.embedder = FixtureEmbedder(256)
        self.vectors = self.embedder.encode([d["text"] for d in self.documents])
        self.index = Index(256, metric="cosine", threads=1)
        for document, vector in zip(self.documents, self.vectors):
            self.index.add(int(document["id"]), vector)
        self.index.build_ivf(min(8, len(self.documents)), iterations=8, seed=42)
        self.mean = self.vectors.mean(axis=0)
        _, _, components = np.linalg.svd(self.vectors - self.mean, full_matrices=False)
        self.components = components[:2].T
        coordinates = (self.vectors - self.mean) @ self.components
        self.scale = max(float(np.abs(coordinates).max()), 1e-6)
        self.points = [
            {
                "id": d["id"],
                "x": float(x / self.scale),
                "y": float(y / self.scale),
                "source": d["source"],
            }
            for d, (x, y) in zip(self.documents, coordinates)
        ]

    def bootstrap(self):
        return {
            "engine": {
                "version": __version__,
                "native": True,
                "avx2": bool(avx2_available()),
            },
            "collection": {
                "name": "Vortex documentation",
                "vectors": len(self.index),
                "dimensions": self.index.dimension,
                "nlist": self.index.nlist,
                "metric": "cosine",
                "representation": "FP32",
                "storage": "In-memory",
                "sources": len({d["source"] for d in self.documents}),
                "embedder": self.embedder.name,
            },
            "documents": self.documents,
            "projection": self.points,
            "projection_method": "2D PCA of token-hash fixture vectors",
            "suggestions": [
                "How do memory-mapped snapshots work?",
                "How does IVF search find neighbors?",
                "What does INT8 retain for reranking?",
                "What was measured at ten million vectors?",
            ],
        }

    def search(self, payload):
        if not isinstance(payload, dict):
            raise TypeError("Send a JSON object with a query")
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Enter a query between 1 and 2,000 characters")
        mode, k, nprobe = (
            payload.get("mode", "exact"),
            payload.get("k", 5),
            payload.get("nprobe", 4),
        )
        if not isinstance(mode, str) or mode not in {"exact", "ivf"}:
            raise ValueError("Choose exact or IVF search")
        if type(k) is not int or not 1 <= k <= min(20, len(self.documents)):
            raise ValueError("Top k must be an integer from 1 to 20")
        if type(nprobe) is not int or not 1 <= nprobe <= self.index.nlist:
            raise ValueError(f"Probes must be between 1 and {self.index.nlist}")
        started = time.perf_counter()
        try:
            vector = self.embedder.encode([query.strip()])[0]
        except ValueError as error:
            raise ValueError(
                "Add a descriptive search term, such as snapshot or cache"
            ) from error
        embedded = time.perf_counter()
        found = self.index.search(vector, k=k, nprobe=nprobe if mode == "ivf" else 0)
        searched = time.perf_counter()
        truth = found if mode == "exact" else self.index.search(vector, k=k, nprobe=0)
        expected = {hit[0] for hit in truth["hits"]}
        recall = len(expected & {hit[0] for hit in found["hits"]}) / len(expected)
        point = (vector - self.mean) @ self.components / self.scale
        return {
            "query": query.strip(),
            "mode": mode,
            "k": k,
            "nprobe": nprobe if mode == "ivf" else 0,
            "hits": [
                {**self.lookup[int(identifier)], "score": float(score), "rank": rank}
                for rank, (identifier, score) in enumerate(found["hits"], 1)
            ],
            "scanned": found["scanned"],
            "total_vectors": len(self.index),
            "recall_at_k": recall,
            "native_ms": found["milliseconds"],
            "embedding_ms": (embedded - started) * 1000,
            "search_wall_ms": (searched - embedded) * 1000,
            "query_point": {"x": float(point[0]), "y": float(point[1])},
            "embedding": "Local token-hash fixture, not a pretrained semantic model",
        }

    def benchmarks(self):
        base = self.root / "benchmarks/reference/capacity"
        summaries = json.loads((base / "summary.json").read_text(encoding="utf-8"))
        diagnostics = json.loads(
            (base / "timing_diagnostics.json").read_text(encoding="utf-8")
        )
        runs = []
        for row in summaries:
            result = json.loads(
                (base / str(row["vectors"]) / "result.json").read_text(encoding="utf-8")
            )
            runs.append(
                {
                    **row,
                    "exact": result["exact"],
                    "probes": result["ivf"],
                    "snapshot_sha256": result["snapshot_sha256"],
                    "query_count": result["query_count"],
                    "repeats": result["repeats"],
                }
            )
        return {
            "kind": "recorded",
            "recorded_date": "2026-11-11",
            "dimensions": 64,
            "metric": "squared L2",
            "runs": runs,
            "diagnostics": diagnostics["runs"],
            "scope": "Synthetic 64D FP32; 20 queries, three rounds; requested 1536 MiB resident working set. These are archived measurements, separate from the live documentation collection.",
        }


def make_server(app, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, body, content_type="application/json"):
            data = (
                json.dumps(body, allow_nan=False).encode()
                if content_type == "application/json"
                else body
            )
            self.send_response(status)
            self.send_header(
                "Content-Type",
                content_type
                + ("; charset=utf-8" if content_type != "image/svg+xml" else ""),
            )
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            allowed = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            if self.server.server_port == 80:
                allowed.update({"127.0.0.1", "localhost"})
            return self.headers.get("Host") in allowed

        def reject_post(self, status, message):
            # A close with unread kernel-buffered body data can turn the error
            # response into a TCP reset on Windows. Drain only a bounded amount;
            # rejected input is never parsed, and a slow sender cannot hold this
            # worker indefinitely.
            try:
                remaining = min(
                    max(0, int(self.headers.get("Content-Length", "0"))), 65536
                )
            except ValueError:
                remaining = 0
            self.close_connection = True
            deadline = time.monotonic() + 2
            try:
                while remaining:
                    budget = deadline - time.monotonic()
                    if budget <= 0:
                        break
                    self.connection.settimeout(budget)
                    block = self.rfile.read1(min(4096, remaining))
                    if not block:
                        break
                    remaining -= len(block)
            except OSError:
                pass
            self.connection.settimeout(2)
            self.respond(status, {"error": message})

        def do_GET(self):
            if not self.valid_host():
                self.respond(403, {"error": "Use the local workbench address"})
                return
            route = unquote(urlsplit(self.path).path)
            if route == "/api/bootstrap":
                self.respond(200, app.bootstrap())
            elif route == "/api/benchmarks":
                try:
                    self.respond(200, app.benchmarks())
                except (OSError, ValueError, KeyError):
                    self.respond(
                        503, {"error": "Recorded benchmark evidence is unavailable"}
                    )
            elif route in STATIC:
                filename, kind = STATIC[route]
                self.respond(200, (app.root / "frontend" / filename).read_bytes(), kind)
            elif route == "/architecture.svg":
                self.respond(
                    200,
                    (app.root / "assets/system-architecture.svg").read_bytes(),
                    "image/svg+xml",
                )
            elif route.removeprefix("/source/") in app.sources and route.startswith(
                "/source/"
            ):
                self.respond(
                    200,
                    (app.root / route.removeprefix("/source/")).read_bytes(),
                    "text/plain",
                )
            else:
                self.respond(404, {"error": "Not found"})

        def do_POST(self):
            if not self.valid_host():
                self.reject_post(403, "Use the local workbench address")
                return
            origin = self.headers.get("Origin")
            if origin and origin not in {
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            }:
                self.reject_post(403, "Search requests must come from this workbench")
                return
            if urlsplit(self.path).path != "/api/search":
                self.reject_post(404, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    self.reject_post(413, "Search request is too large or empty")
                    return
                if (
                    self.headers.get("Content-Type", "").split(";")[0]
                    != "application/json"
                ):
                    self.reject_post(415, "Send application/json")
                    return
                payload = json.loads(self.rfile.read(length))
                self.respond(200, app.search(payload))
            except (TypeError, ValueError, UnicodeDecodeError) as error:
                self.respond(400, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be from 0 to 65535")
    server = make_server(Workbench(), args.port)
    print(f"Vortex workbench: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
