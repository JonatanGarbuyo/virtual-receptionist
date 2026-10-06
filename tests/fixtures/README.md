# Audio fixtures (#24)

`*.pcm` probes are **generated, never committed** (see
`NoAudioPersistenceTest`, which forbids audio files anywhere in the
repo). Every fixture is synthesized locally by
`generate_fixtures.py` (deterministic tones/silence, signed PCM16
mono). Provenance: project-owned synthetic signals, no license
encumbrance, no speaker identity, no customer audio.

Regenerate: `PYTHONPATH=src python3 tests/fixtures/generate_fixtures.py`

Real-speech STT validation never uses these tones (a tone is not
speech): `tests/test_cascaded_real.py` accepts an operator-provided
Spanish fixture via `REAL_AUDIO_FIXTURE` and skips otherwise.
