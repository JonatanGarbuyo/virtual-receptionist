"""Deterministic transfer/fallback policy over configured symbolic data.

The model/requester only ever names a symbolic destination id. This
engine resolves it to a trusted PBX-local target, enforces attempt
budgets and limits, and classifies every refusal fail-closed. It holds
plain data (seeded from configuration in production, from literals in
tests); persistence of SQLite records belongs to a later ticket.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Destination:
    """One configured symbolic destination and its trusted PBX-local target."""

    id: str
    target: str
    kind: str
    enabled: bool = True


@dataclass(frozen=True)
class Limits:
    """v0.1 default limits from the spec; all configurable via construction."""

    max_call_seconds: float = 600.0
    max_transfer_attempts: int = 3
    max_turns: int = 30
    message_capture_seconds: float = 90.0


class DestinationStatus(Enum):
    OK = "ok"
    UNKNOWN = "unknown"
    DISABLED = "disabled"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class Resolution:
    status: DestinationStatus
    destination: Destination | None = None


_SYMBOLIC_ID = re.compile(r"[A-Za-z0-9_-]+")


class PolicyEngine:
    """Resolves symbolic ids and enforces transfer budgets and limits."""

    def __init__(
        self,
        destinations: dict[str, Destination],
        fallback_id: str,
        limits: Limits,
    ) -> None:
        self._destinations = dict(destinations)
        self._fallback_id = fallback_id
        self._limits = limits

    @property
    def limits(self) -> Limits:
        return self._limits

    def resolve(self, destination_id: object) -> Resolution:
        """Classify one requested id fail-closed. Never raises on input."""
        if not isinstance(destination_id, str):
            return Resolution(DestinationStatus.MALFORMED)
        if not destination_id or _SYMBOLIC_ID.fullmatch(destination_id) is None:
            return Resolution(DestinationStatus.MALFORMED)
        destination = self._destinations.get(destination_id)
        if destination is None:
            return Resolution(DestinationStatus.UNKNOWN)
        if not destination.enabled:
            return Resolution(DestinationStatus.DISABLED)
        return Resolution(DestinationStatus.OK, destination)

    def fallback(self) -> Destination | None:
        """The configured symbolic fallback, or None when unusable."""
        resolution = self.resolve(self._fallback_id)
        if resolution.status is not DestinationStatus.OK:
            return None
        return resolution.destination

    def transfer_allowed(self, attempts_used: int) -> bool:
        """Whether another telephony transfer attempt fits the budget."""
        return attempts_used < self._limits.max_transfer_attempts
