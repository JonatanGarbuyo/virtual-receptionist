"""Project-owned telephony boundary used by the baresip-python spike.

This file deliberately contains no baresip types.  A future production adapter
may be in-process or out-of-process without changing the receptionist core.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Protocol, TypeAlias


CallId: TypeAlias = str
DestinationId: TypeAlias = str


@dataclass(frozen=True, slots=True)
class ResolvedDestination:
    """Trusted PBX-local target resolved by Policy/Config, never by the model."""

    id: DestinationId
    sip_target: str


class TelephonyEventKind(str, Enum):
    REGISTERED = "registered"
    REGISTRATION_FAILED = "registration_failed"
    INCOMING_CALL = "incoming_call"
    CALL_ESTABLISHED = "call_established"
    CALL_CLOSED = "call_closed"
    DTMF = "dtmf"
    REMOTE_HOLD = "remote_hold"
    REMOTE_RESUME = "remote_resume"
    TRANSPORT_ERROR = "transport_error"


@dataclass(frozen=True, slots=True)
class TelephonyEvent:
    kind: TelephonyEventKind
    call_id: CallId | None = None
    peer: str | None = None
    digit: str | None = None
    detail: str | None = None


class TransferResult(str, Enum):
    ACCEPTED_BY_PBX = "accepted_by_pbx"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    TRANSPORT_ERROR = "transport_error"


class TelephonyPort(Protocol):
    """Minimal contract the receptionist core is allowed to see.

    Notice the deliberate absence of a generic dial(uri) operation.
    """

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def events(self) -> AsyncIterator[TelephonyEvent]: ...

    async def answer(self, call_id: CallId) -> None: ...

    async def reject(self, call_id: CallId) -> None: ...

    async def hangup(self, call_id: CallId) -> None: ...

    async def hold(self, call_id: CallId) -> None: ...

    async def resume(self, call_id: CallId) -> None: ...

    async def transfer(
        self, call_id: CallId, destination: ResolvedDestination
    ) -> TransferResult: ...

    async def send_dtmf(self, call_id: CallId, digits: str) -> None: ...

    def read_pcm(self, call_id: CallId, max_bytes: int = 4096) -> bytes: ...

    def write_pcm(self, call_id: CallId, pcm: bytes) -> int: ...
