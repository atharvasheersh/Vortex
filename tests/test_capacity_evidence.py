"""Check the published large-collection evidence and retained diagnostics."""

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from validate_capacity import validate_archive


def test_published_capacity_archive_is_consistent():
    root = ROOT / "benchmarks" / "reference" / "capacity"
    outcome = validate_archive(root)
    assert {r["size"] for r in outcome["runs"]} == {1000000, 10000000}
    assert all(r["records"] == 480 for r in outcome["runs"])
    gaps = [gap for r in outcome["runs"] for gap in r["timing_gaps"]]
    assert len(gaps) == 1 and gaps[0]["nprobe"] == 32
    assert gaps[0]["wall_ms"] > gaps[0]["process_cpu_seconds"] * 100000
    assert not list(root.rglob("*.vtx"))
    attempts = [
        json.loads(p.read_text()) for p in (root / "attempts").glob("*/result.json")
    ]
    assert len(attempts) == 2
    assert all(r["status"] == "resource_limited" for r in attempts)


def test_modified_capacity_evidence_fails_validation(tmp_path):
    source = ROOT / "benchmarks" / "reference" / "capacity"
    archive = tmp_path / "archive"
    shutil.copytree(source, archive)
    with (archive / "10000000" / "measurements.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        validate_archive(archive)
