"""Versioned evaluation contracts and reproducibility manifests."""
import hashlib
import ctypes
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def process_memory():
    """Process-wide high-water marks include Python, temporary arrays, and baselines."""
    if sys.platform == "win32":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb",wintypes.DWORD),("page_faults",wintypes.DWORD)] + [
                (name,ctypes.c_size_t) for name in ("peak_working_set","working_set","peak_paged_pool",
                 "paged_pool","peak_nonpaged_pool","nonpaged_pool","pagefile","peak_pagefile")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi",use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            return {"peak_rss_bytes":counters.peak_working_set,"current_rss_bytes":counters.working_set,
                    "method":"GetProcessMemoryInfo; process-wide working set"}
        return {"peak_rss_bytes":None,"method":"GetProcessMemoryInfo unavailable"}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_rss_bytes":int(peak if sys.platform == "darwin" else peak*1024),
            "method":"getrusage RUSAGE_SELF; process-wide peak RSS"}


def load_dataset(path):
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    if content.get("schema_version") != 1:
        raise ValueError("unsupported evaluation schema version")
    chunks, cases = content["chunks"], content["cases"]
    ids = [chunk["id"] for chunk in chunks]
    if any(type(identifier) is not int or identifier < 0 or identifier >= 2 ** 64 for identifier in ids):
        raise ValueError("chunk IDs must be uint64 integers")
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("chunk IDs must be unique and nonempty")
    if any(not isinstance(chunk["text"], str) or not chunk["text"].strip() for chunk in chunks):
        raise ValueError("chunk text required")
    seen = set()
    if not cases:
        raise ValueError("evaluation cases required")
    for case in cases:
        if not isinstance(case["query_id"], str) or not case["query_id"] or case["query_id"] in seen:
            raise ValueError("query IDs must be unique nonempty strings")
        seen.add(case["query_id"])
        relevant = case["relevant_chunk_ids"]
        if (not isinstance(case["query"], str) or not case["query"].strip() or not relevant
                or any(type(identifier) is not int for identifier in relevant)
                or len(set(relevant)) != len(relevant) or not set(relevant).issubset(ids)):
            raise ValueError("invalid query or relevant chunk IDs")
        if "reference_answer" in case and case["reference_answer"] is not None and not isinstance(case["reference_answer"], str):
            raise ValueError("reference_answer must be a string or null")
    return content


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    if not (root / "CMakeLists.txt").is_file():
        root = Path.cwd()
    paths = [root / name for name in ("CMakeLists.txt", "pyproject.toml")]
    for directory in ("include", "src", "python", "eval", "benchmarks"):
        paths.extend(path for path in (root/directory).rglob("*")
                     if path.suffix in (".cpp", ".hpp", ".py") and "__pycache__" not in path.parts)
    digest = hashlib.sha256()
    found = False
    for path in sorted(paths):
        if path.is_file():
            found = True
            digest.update(path.relative_to(root).as_posix().encode()+b"\0"+path.read_bytes()+b"\0")
    return digest.hexdigest() if found else None


def manifest(configuration, dataset_path=None):
    def git(*args):
        try:
            result = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=normal")
    packages = {}
    for package in ("numpy", "pybind11", "faiss-cpu", "sentence-transformers", "vortex-retrieval"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "working_tree_dirty": bool(status) if status is not None else None,
        "source_sha256": source_fingerprint(),
        "dataset_sha256": fingerprint(dataset_path) if dataset_path else None,
        "configuration": configuration,
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "processor": platform.processor(), "python": sys.version},
        "packages": packages, "process_memory": process_memory(),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
