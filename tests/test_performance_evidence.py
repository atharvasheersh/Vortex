"""Reject changed optimization evidence even when its inventory is rehashed."""

import gzip
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from report_performance import sha256, write_json
from validate_performance import validate

ARCHIVE = ROOT / "benchmarks/reference/performance"


def test_published_optimization_evidence_is_self_consistent():
    result = validate(ARCHIVE)
    assert result["records"] == 8924
    assert result["tuning_records"] == 768
    assert result["conditions"] == 20
    assert result["optional_records"] == 64


@pytest.mark.parametrize("change", ["summary", "memory", "query_split", "result"])
def test_rehashed_optimization_tampering_is_rejected(change, tmp_path):
    target = tmp_path / "evidence"
    shutil.copytree(ARCHIVE, target)
    checksums = json.loads((target / "checksums.json").read_text())
    if change == "summary":
        name = "summary.json"
        data = json.loads((target / name).read_text())
        data[0]["p95_ms"] *= 0.5
        write_json(target / name, data)
    elif change == "memory":
        name = "manifests.json"
        data = json.loads((target / name).read_text())
        next(iter(data.values()))["working_set_mib"] = 4096
        write_json(target / name, data)
    else:
        name = "measurements.jsonl.gz"
        data = [
            json.loads(line)
            for line in gzip.decompress((target / name).read_bytes())
            .decode()
            .splitlines()
        ]
        if change == "query_split":
            data[0]["query_id"] = 0
        else:
            data[0]["hits"][0][0] = 2**63
        (target / name).write_bytes(
            gzip.compress(
                ("".join(json.dumps(row) + "\n" for row in data)).encode(), mtime=0
            )
        )
    checksums[name] = sha256(target / name)
    write_json(target / "checksums.json", checksums)
    with pytest.raises(ValueError):
        validate(target)
