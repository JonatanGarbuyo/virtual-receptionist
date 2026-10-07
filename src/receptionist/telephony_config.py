"""Telephony configuration domain (#25): declarative SIP identity.

Canonical store is config.db through ConfigService (flat ``sip.*`` keys,
never a giant JSON blob, never env vars as production config). Secrets
are ``repr=False`` and are never logged: only the username/domain/codec
selection may appear in diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SIP_USERNAME_KEY = "sip.username"
SIP_AUTH_USER_KEY = "sip.auth_user"
SIP_PASSWORD_KEY = "sip.password"  # nosec: key name only, never a value
SIP_DOMAIN_KEY = "sip.domain"
SIP_REGISTRAR_KEY = "sip.registrar"
SIP_DISPLAY_NAME_KEY = "sip.display_name"
SIP_TRANSPORT_KEY = "sip.transport"
SIP_REG_INTERVAL_KEY = "sip.reg_interval"
SIP_CODECS_KEY = "sip.codecs"
SIP_DTMF_MODE_KEY = "sip.dtmf_mode"
SIP_BIND_KEY = "sip.bind"
SIP_MAX_CALLS_KEY = "sip.max_calls"
SIP_RTP_TIMEOUT_KEY = "sip.rtp_timeout"
SIP_EXTRA_CONFIG_KEY = "sip.extra_config"

_VALID_TRANSPORTS = ("udp", "tcp", "tls")
_VALID_DTMF_MODES = ("rtpevent", "info", "auto")
_VALID_CODECS = ("pcmu", "pcma")


@dataclass(frozen=True)
class TelephonyConfig:
    """Declarative SIP endpoint identity for one installation.

    ``username``/``domain`` form ``sip:username@domain``; ``registrar``
    is the outbound proxy when it differs from the domain. ``auth_user``
    overrides the digest username only for credential-list deployments.
    ``audio_codecs`` is the preference-ordered allowlist (subset of PCMU/
    PCMA in v0.1). ``bind`` pins the local interface (required for
    loopback benches); ``None`` uses normal interface discovery.
    """

    username: str
    domain: str
    password: str = field(repr=False, default="")
    auth_user: str | None = None
    registrar: str | None = None
    display_name: str = ""
    transport: str = "udp"
    reg_interval: int = 600
    audio_codecs: tuple[str, ...] = ("pcmu", "pcma")
    dtmf_mode: str = "rtpevent"
    bind: str | None = None
    max_calls: int = 2
    rtp_timeout: int = 0
    extra_config_text: str = ""
    sip_trace: bool = False

    def __post_init__(self) -> None:
        if not self.username or not self.username.strip():
            raise ValueError("sip.username must not be empty")
        if not self.domain or not self.domain.strip():
            raise ValueError("sip.domain must not be empty")
        if self.transport not in _VALID_TRANSPORTS:
            raise ValueError(f"sip.transport must be one of {_VALID_TRANSPORTS}")
        if self.dtmf_mode not in _VALID_DTMF_MODES:
            raise ValueError(f"sip.dtmf_mode must be one of {_VALID_DTMF_MODES}")
        if not self.audio_codecs:
            raise ValueError("sip.codecs must allow at least one codec")
        for codec in self.audio_codecs:
            if codec not in _VALID_CODECS:
                raise ValueError(f"unsupported sip codec {codec!r}")
        if self.reg_interval < 0:
            raise ValueError("sip.reg_interval must be >= 0")
        if self.max_calls < 1:
            raise ValueError("sip.max_calls must be >= 1")

    @property
    def redacted_summary(self) -> str:
        """Log-safe one-liner: identity without secrets."""
        return (
            f"sip:{self.username}@{self.domain} "
            f"transport={self.transport} "
            f"codecs={','.join(self.audio_codecs)} "
            f"dtmf={self.dtmf_mode}"
        )


def telephony_config_from_mapping(get: object) -> TelephonyConfig:
    """Build a :class:`TelephonyConfig` from a ``key -> str`` lookup.

    ``get`` is any callable ``(key) -> str | None`` (ConfigRepository.get
    in production, a dict lookup in the integration harness which may
    seed values from environment variables for test orchestration only).
    Only explicit ``sip.*`` keys are read.
    """

    def raw(key: str) -> str:
        value = get(key)  # type: ignore[operator]
        return (value or "").strip() if isinstance(value, str) else ""

    def _int(key: str, default: int) -> int:
        text = raw(key)
        if not text:
            return default
        try:
            value = int(text)
        except ValueError:
            return default
        return value

    codecs_raw = raw(SIP_CODECS_KEY) or "pcmu,pcma"
    codecs = tuple(
        part.strip().lower() for part in codecs_raw.split(",") if part.strip()
    ) or ("pcmu", "pcma")

    return TelephonyConfig(
        username=raw(SIP_USERNAME_KEY),
        domain=raw(SIP_DOMAIN_KEY),
        password=raw(SIP_PASSWORD_KEY),
        auth_user=raw(SIP_AUTH_USER_KEY) or None,
        registrar=raw(SIP_REGISTRAR_KEY) or None,
        display_name=raw(SIP_DISPLAY_NAME_KEY),
        transport=(raw(SIP_TRANSPORT_KEY) or "udp").lower(),
        reg_interval=_int(SIP_REG_INTERVAL_KEY, 600),
        audio_codecs=codecs,  # type: ignore[arg-type]
        dtmf_mode=(raw(SIP_DTMF_MODE_KEY) or "rtpevent").lower(),
        bind=raw(SIP_BIND_KEY) or None,
        max_calls=_int(SIP_MAX_CALLS_KEY, 2),
        rtp_timeout=_int(SIP_RTP_TIMEOUT_KEY, 0),
        extra_config_text=raw(SIP_EXTRA_CONFIG_KEY),
    )
