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

#: Canonical voice-profile keys in config.db. Structural values (model,
#: runtime executable, profile, voice) require a backend restart/reload;
#: a live call keeps the snapshot it started with.
VOICE_PROFILE_KEY = "voice.profile"
VOICE_MODEL_ROOT_KEY = "voice.model_root"
VOICE_MANIFEST_KEY = "voice.manifest"
VOICE_STT_EXECUTABLE_KEY = "voice.stt_executable"
VOICE_LLM_EXECUTABLE_KEY = "voice.llm_executable"
VOICE_TTS_VOICE_KEY = "voice.tts_voice"
VOICE_TTS_SPEAKER_KEY = "voice.tts_speaker_id"
VOICE_MAX_CONTEXT_CHARS_KEY = "voice.max_context_chars"
VOICE_MAX_SPOKEN_CHARS_KEY = "voice.max_spoken_chars"


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

    def voice_profile(self) -> VoiceProfile:
        """Typed voice-profile read over config.db. Only the small
        operator-configurable domain: profile id, trusted model root /
        manifest, voice selection, context/output bounds, and runtime
        executable paths. No workflow KV, no env-var second source."""
        from receptionist.cascaded import VoiceProfile

        get = self._repository.get

        def _positive_int(key: str, default: int) -> int:
            raw = (get(key) or "").strip()
            if not raw:
                return default
            try:
                value = int(raw)
            except ValueError:
                return default
            return value if value > 0 else default

        def _non_negative_int(key: str, default: int) -> int:
            raw = (get(key) or "").strip()
            if not raw:
                return default
            try:
                value = int(raw)
            except ValueError:
                return default
            return value if value >= 0 else default

        return VoiceProfile(
            profile_id=(get(VOICE_PROFILE_KEY) or "cascaded-cpu-baseline-v1").strip()
            or "cascaded-cpu-baseline-v1",
            model_root=(get(VOICE_MODEL_ROOT_KEY) or "").strip(),
            manifest_path=(get(VOICE_MANIFEST_KEY) or "").strip(),
            stt_executable=(get(VOICE_STT_EXECUTABLE_KEY) or "").strip(),
            llm_executable=(get(VOICE_LLM_EXECUTABLE_KEY) or "").strip(),
            tts_voice=(get(VOICE_TTS_VOICE_KEY) or "es-female-1").strip()
            or "es-female-1",
            tts_speaker_id=_non_negative_int(VOICE_TTS_SPEAKER_KEY, 0),
            max_context_chars=_positive_int(VOICE_MAX_CONTEXT_CHARS_KEY, 9000),
            max_spoken_chars=_positive_int(VOICE_MAX_SPOKEN_CHARS_KEY, 500),
        )
