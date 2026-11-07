"""Check published evidence and detect tampering before reporting measurements."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "benchmarks" / "validate_scaling.py"


def test_published_scale_evidence_is_consistent():
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--directory",
            str(ROOT / "benchmarks" / "reference" / "scaling"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 sizes verified" in result.stdout


def test_modified_evidence_fails_checksum_validation(tmp_path):
    source = ROOT / "benchmarks" / "reference" / "scaling"
    target = tmp_path / "archive"
    shutil.copytree(source, target)
    with (target / "1000" / "result.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--directory", str(target)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "checksum mismatch" in result.stderr
