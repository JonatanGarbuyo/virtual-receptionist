"""Canonical configuration behind a service/repository boundary.

Callers read configuration through :class:`ConfigService`, never through
scattered direct SQL. The repository implementation (in-memory here,
SQLite in a later ticket) stays behind the boundary.
"""

from __future__ import annotations

from receptionist.boundaries import ConfigRepository

GREETING_KEY = "greeting"
LANGUAGE_KEY = "language"

REQUIRED_KEYS = (GREETING_KEY, LANGUAGE_KEY)


class InMemoryConfigRepository:
    """Stand-in repository seeded from a plain dict. SQLite arrives later."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = dict(values)

    def get(self, key: str) -> str | None:
        return self._values.get(key)


class ConfigService:
    """Typed reads over a :class:`ConfigRepository`."""

    def __init__(self, repository: ConfigRepository) -> None:
        self._repository = repository

    def get_greeting(self) -> str | None:
        return self._repository.get(GREETING_KEY)

    def get_language(self) -> str | None:
        return self._repository.get(LANGUAGE_KEY)

    def missing_required(self) -> list[str]:
        """Required keys that are absent or blank."""
        missing = []
        for key in REQUIRED_KEYS:
            value = self._repository.get(key)
            if value is None or not value.strip():
                missing.append(key)
        return missing
