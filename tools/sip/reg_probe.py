"""Negative registration probe for #25 (own process: one Runtime each).

Usage: reg_probe.py USER PASSWORD DOMAIN REGISTRAR
Starts a BaresipTelephonyAdapter, prints the resulting registration
state, shuts down cleanly. Exit 0 always (the state line is the result).
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from receptionist.baresip_adapter import BaresipTelephonyAdapter
from receptionist.telephony_config import TelephonyConfig


def main() -> int:
    user, password, domain, registrar = sys.argv[1:5]
    adapter = BaresipTelephonyAdapter(
        TelephonyConfig(
            username=user,
            domain=domain,
            password=password,
            registrar=registrar,
            bind="127.0.0.1",
            reg_interval=600,
            extra_config_text="sip_listen 127.0.0.1:5098\n",
        )
    )
    state = adapter.start(timeout=12.0)
    print(f"STATE={state.value}", flush=True)
    adapter.shutdown()
    print("SHUTDOWN_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
