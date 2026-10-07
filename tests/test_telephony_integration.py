"""Gated tier-3 integration: SIPp + disposable Asterisk + baresip (#25).

Skipped by default (needs docker + pinned images + loopback SIP/RTP).
Run explicitly:

    VR_RUN_SIP=1 python -m unittest tests.test_telephony_integration -v

The matrix runner (tools/sip/run_matrix.py) executes the full telephony
matrix against real software and writes
docs/evidence/25-baresip-matrix/evidence.json. This test fails when any
matrix check fails; the JSON is the evidence, not the assertion text.
PR CI never runs this (fast deterministic suites only).
"""

import json
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE = os.path.join(
    REPO, "docs", "evidence", "25-baresip-matrix", "evidence.json"
)
RUN_MATRIX = os.path.join(REPO, "tools", "sip", "run_matrix.py")

RUN_SIP = os.getenv("VR_RUN_SIP", "") == "1"
MATRIX_CODECS = os.getenv("VR_SIP_CODECS", "pcmu")


@unittest.skipUnless(RUN_SIP, "needs docker/Asterisk/SIPp (VR_RUN_SIP=1)")
class TelephonyMatrixTest(unittest.TestCase):
    def test_real_sip_matrix(self) -> None:
        """Run the matrix; every check must pass (see evidence.json)."""
        env = dict(os.environ, PYTHONPATH=f"src{os.pathsep}tests")
        proc = subprocess.run(
            [sys.executable, RUN_MATRIX, "--codecs", MATRIX_CODECS],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        try:
            with open(EVIDENCE) as handle:
                evidence = json.load(handle)
        except (OSError, ValueError) as error:
            self.fail(f"no evidence written: {error}\n{proc.stdout[-2000:]}")
        failed = [
            item for item in evidence.get("checks", []) if item.get("result") != "pass"
        ]
        summary = "\n".join(
            f"FAILED {item['id']}: {item.get('detail', '')}" for item in failed
        )
        self.assertEqual(
            failed,
            [],
            f"matrix failed ({len(failed)} checks):\n{summary}\n"
            f"runner rc={proc.returncode}",
        )


if __name__ == "__main__":
    unittest.main()
