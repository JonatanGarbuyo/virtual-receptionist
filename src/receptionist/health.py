"""Service health with STARTING / READY / DEGRADED / NOT_READY semantics."""

from __future__ import annotations

from enum import Enum


class HealthStatus(Enum):
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    NOT_READY = "not_ready"


class ServiceHealth:
    def __init__(self, detail: str = "initializing") -> None:
        self.status = HealthStatus.STARTING
        self.detail = detail

    def mark_ready(self, detail: str = "ready") -> None:
        self.status = HealthStatus.READY
        self.detail = detail

    def mark_degraded(self, detail: str) -> None:
        self.status = HealthStatus.DEGRADED
        self.detail = detail

    def mark_not_ready(self, detail: str) -> None:
        self.status = HealthStatus.NOT_READY
        self.detail = detail

    def recover(self, detail: str = "ready") -> None:
        if self.status is HealthStatus.DEGRADED:
            self.mark_ready(detail)
