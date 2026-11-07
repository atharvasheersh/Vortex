"""Check the public runner's execution and evidence-preservation contract."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "benchmarks" / "run_scaling.py"


@pytest.mark.parametrize(
    "arguments",
    [["--dimension", "0"], ["--queries", "0"], ["--nlist", "10001"], ["--sizes", "1"]],
)
def test_invalid_configuration_is_rejected_before_execution(arguments, tmp_path):
    output = tmp_path / "measurements"
    result = subprocess.run(
        [sys.executable, str(RUNNER), *arguments, "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert not output.exists()


def test_existing_evidence_is_preserved(tmp_path):
    folder = tmp_path / "64"
    folder.mkdir()
    sentinel = folder / "result.json"
    sentinel.write_text('{"preserve": true}', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--sizes",
            "64",
            "--nlist",
            "4",
            "--output",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert sentinel.read_text(encoding="utf-8") == '{"preserve": true}'


def test_small_isolated_run_emits_consistent_results(tmp_path):
    pytest.importorskip("faiss")
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--sizes",
            "64",
            "--dimension",
            "8",
            "--nlist",
            "4",
            "--queries",
            "6",
            "--repeats",
            "1",
            "--k",
            "5",
            "--output",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads((tmp_path / "64" / "result.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "passed"
    assert evidence["faiss_flat"]["exact_neighbor_agreement"] == 1
    assert len(evidence["numpy_checked_query_ids"]) == 3
    assert evidence["selected_ivf"]["recall_at_k"] >= 0.95
    assert evidence["ivf"][-1]["nprobe"] == 4
    assert evidence["ivf"][-1]["recall_at_k"] == 1
    assert evidence["mapped_ivf"]["cache_stats"]["resident_bytes"] <= 8 * 1024 * 1024
    assert (
        evidence["snapshot_bytes"]
        == (tmp_path / "64" / "collection.vtx").stat().st_size
    )
    assert "6 unique" in evidence["manifest"]["configuration"]["measurement"]
