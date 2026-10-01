"""Call id generation.

Production ids must stay unique across process restarts so a new call can
never overwrite persisted history (SQLite PRIMARY KEY). UUIDs give that
without any durable sequence or database round-trip.
"""

from __future__ import annotations

import uuid


class UuidCallIds:
    """Production generator: random, restart-safe ids."""

    def next_id(self) -> str:
        return f"call-{uuid.uuid4().hex[:12]}"
