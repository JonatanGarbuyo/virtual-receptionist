"""Canonical configuration behind a service/repository boundary.

Callers read configuration through :class:`ConfigService`, never through
scattered direct SQL. The repository implementation (in-memory here,
SQLite in a later ticket) stays behind the boundary.
"""

from __future__ import annotations

from receptionist.alerting import AlertConfigRepository, AlertSettings
from receptionist.boundaries import (
    ConfigRepository,
    KnowledgeSourceDeclaration,
    KnowledgeSourceRepository,
)

GREETING_KEY = "greeting"
LANGUAGE_KEY = "language"
TRANSCRIPTS_ENABLED_KEY = "transcripts_enabled"

REQUIRED_KEYS = (GREETING_KEY, LANGUAGE_KEY)


class InMemoryConfigRepository:
    """Stand-in repository seeded from a plain dict. SQLite arrives later."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = dict(values)

    def get(self, key: str) -> str | None:
        return self._values.get(key)


class ConfigService:
    """Typed reads over a :class:`ConfigRepository`.

    Knowledge source declarations ride along when a config-side
    :class:`KnowledgeSourceRepository` is provided (same config.db);
    without one no knowledge source is declared. Alert settings ride
    along the same way through an alert repository.
    """

    def __init__(
        self,
        repository: ConfigRepository,
        knowledge_sources: KnowledgeSourceRepository | None = None,
        alerts: AlertConfigRepository | None = None,
    ) -> None:
        self._repository = repository
        self._knowledge_sources = knowledge_sources
        self._alerts = alerts

    def get_greeting(self) -> str | None:
        return self._repository.get(GREETING_KEY)

    def get_language(self) -> str | None:
        return self._repository.get(LANGUAGE_KEY)

    def transcripts_enabled(self) -> bool:
        """Persistent full transcripts. Off unless explicitly enabled."""
        return (self._repository.get(TRANSCRIPTS_ENABLED_KEY) or "").strip().lower() == "true"

    def missing_required(self) -> list[str]:
        """Required keys that are absent or blank."""
        missing = []
        for key in REQUIRED_KEYS:
            value = self._repository.get(key)
            if value is None or not value.strip():
                missing.append(key)
        return missing

    def knowledge_source_declarations(self) -> list[KnowledgeSourceDeclaration]:
        """Operator-declared knowledge sources from config.db. Empty when
        no knowledge repository is wired; the assembler builds live
        sources only from these trusted declarations."""
        if self._knowledge_sources is None:
            return []
        return self._knowledge_sources.list_all()

    def alert_settings(self) -> AlertSettings:
        """Operator alert settings from config.db. All channels disabled
        when no alert repository is wired."""
        if self._alerts is None:
            return AlertSettings()
        return self._alerts.load()
