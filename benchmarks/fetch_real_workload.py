"""Fetch a pinned-size ANN-Benchmarks artifact through verified HTTP ranges."""

import argparse
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_performance import digest, write_json

FILES = {
    "sift-128-euclidean": (
        525128288,
        '"2bf1ef4c0c3e17031bd64980774eea09-63"',
        "dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984",
    ),
    "glove-100-angular": (
        485413888,
        '"9ef1a69a639809d4ae43efd5e6707fa1-58"',
        "544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4",
    ),
    "deep-image-96-angular": (
        3848008288,
        '"b26706a2cc5c8e32410625728145d805-459"',
        None,
    ),
}


def fetch(name, out, workers=8, chunk=4 * 1024**2, seconds=1500):
    expected, expected_etag, expected_sha = FILES[name]
    url = f"https://ann-benchmarks.com/{name}.hdf5"
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".ranges.partial")
    started = time.monotonic()
    curl = ["curl", "--fail", "--silent", "--show-error"]
    head = subprocess.check_output(
        [*curl, "--head", "--max-time", "25", url], text=True
    )
    headers = {
        key.lower(): value.strip()
        for key, value in (
            line.split(":", 1) for line in head.splitlines() if ":" in line
        )
    }
    etag = headers["etag"]
    if int(headers["content-length"]) != expected or etag != expected_etag:
        raise ValueError("published artifact size or identity changed")
    with partial.open("xb") as file:
        file.truncate(expected)

    def download(start):
        end = min(expected - 1, start + chunk - 1)
        for retry in range(3):
            remaining = seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("download budget exhausted")
            try:
                with tempfile.TemporaryDirectory(
                    dir=out.parent, prefix="range-"
                ) as tmp:
                    block = Path(tmp) / "block"
                    info = subprocess.check_output(
                        [
                            *curl,
                            "--range",
                            f"{start}-{end}",
                            "--max-time",
                            str(min(100, remaining)),
                            "--max-filesize",
                            str(end - start + 1),
                            "--output",
                            str(block),
                            "--write-out",
                            "%{http_code}\n%header{etag}\n%header{content-range}",
                            url,
                        ],
                        text=True,
                    ).splitlines()
                    if info != ["206", etag, f"bytes {start}-{end}/{expected}"]:
                        raise ValueError("server changed range or artifact identity")
                    data = block.read_bytes()
                    if len(data) != end - start + 1:
                        raise OSError("incomplete range")
                    with partial.open("r+b") as dest:
                        dest.seek(start)
                        dest.write(data)
                    return len(data)
            except (OSError, subprocess.CalledProcessError):
                if retry == 2:
                    raise
        raise AssertionError("range retry did not finish")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        total = 0
        for length in pool.map(download, range(0, expected, chunk)):
            total += length
            if total // (64 * 1024**2) > (total - length) // (64 * 1024**2):
                print(f"{name}: {total}/{expected} bytes", flush=True)
    if total != expected or partial.stat().st_size != expected:
        raise ValueError("download coverage mismatch")
    sha = digest(partial)
    if expected_sha is not None and sha != expected_sha:
        raise ValueError("artifact does not match its pinned SHA-256")
    os.link(partial, out)
    partial.unlink()
    write_json(
        out.with_suffix(".download.json"),
        {
            "url": url,
            "etag": etag,
            "bytes": total,
            "sha256": sha,
            "seconds": time.monotonic() - started,
            "method": "verified HTTP ranges",
        },
    )
    print(f"{name}: {total} bytes sha256={sha}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=FILES)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--seconds", type=float, default=1500)
    args = parser.parse_args()
    if args.workers < 1 or not 0 < args.seconds < float("inf"):
        parser.error("workers and seconds must be positive and finite")
    fetch(
        args.name,
        args.output,
        workers=args.workers,
        chunk=args.chunk_mib * 1024**2,
        seconds=args.seconds,
    )


if __name__ == "__main__":
    main()
