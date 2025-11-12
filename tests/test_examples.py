"""Smoke tests for example scripts.

These tests ensure that the example scripts can be imported and run their
main execution paths without raising exceptions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Determine the repo root
ROOT = Path(__file__).resolve().parents[1]


def test_vqe_h2_example_runs() -> None:
    """Test that examples/vqe_h2.py runs successfully."""
    script = ROOT / "examples" / "vqe_h2.py"
    assert script.exists(), f"Example script not found: {script}"

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,  # Should complete in seconds
    )

    assert result.returncode == 0, (
        f"Example script failed with return code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    # Verify expected output is present
    assert "Final estimated ground-state energy" in result.stdout, (
        "Expected output message not found in script output"
    )


def test_hybrid_classifier_example_runs() -> None:
    """Test that examples/hybrid_classifier.py runs successfully."""
    script = ROOT / "examples" / "hybrid_classifier.py"
    assert script.exists(), f"Example script not found: {script}"

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,  # Should complete in seconds
    )

    assert result.returncode == 0, (
        f"Example script failed with return code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    # Verify expected output is present
    assert "Test accuracy" in result.stdout, (
        "Expected output message not found in script output"
    )

