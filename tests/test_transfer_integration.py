"""Gated tier-3 integration: symbolic transfers over real SIP/PBX (#26).

Skipped by default (needs docker + pinned Asterisk image + loopback
SIP/RTP). Run explicitly:

    VR_RUN_SIP_TRANSFER=1 python -m unittest tests.test_transfer_integration -v

The focused runner (tools/sip/run_transfer_matrix.py) executes the five
transfer scenarios against real software and writes
docs/evidence/26-transfer-matrix/evidence.json. This test fails when any
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
    REPO, "docs", "evidence", "26-transfer-matrix", "evidence.json"
)
RUN_MATRIX = os.path.join(REPO, "tools", "sip", "run_transfer_matrix.py")

RUN_SIP_TRANSFER = os.getenv("VR_RUN_SIP_TRANSFER", "") == "1"
MATRIX_CODECS = os.getenv("VR_SIP_CODECS", "pcmu")


@unittest.skipUnless(RUN_SIP_TRANSFER, "needs docker/Asterisk (VR_RUN_SIP_TRANSFER=1)")
class TransferMatrixTest(unittest.TestCase):
    def test_real_transfer_matrix(self) -> None:
        """Run the transfer matrix; the runner must exit 0 and pass."""
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
            self.fail(
                f"no evidence written (runner rc={proc.returncode}): "
                f"{error}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        failed = [
            item for item in evidence.get("checks", []) if item.get("result") != "pass"
        ]
        summary = "\n".join(
            f"FAILED {item['id']}: {item.get('detail', '')}" for item in failed
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"transfer matrix runner failed (rc={proc.returncode}):\n{summary}\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}",
        )
        self.assertEqual(
            failed,
            [],
            f"transfer matrix failed ({len(failed)} checks):\n{summary}",
        )


if __name__ == "__main__":
    unittest.main()
