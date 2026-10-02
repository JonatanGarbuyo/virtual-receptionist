"""Local business knowledge: deterministic retrieval behind one service.

The application consumes only ``KnowledgeService``. Sources (structured
FAQ, Markdown/text files) and the retriever (local keyword matching) stay
behind that contract and are replaceable without touching the core.

Security posture: retrieved chunks are informational context only. Nothing
here parses privileged actions from text, resolves destinations, or acts.
The source list always comes from trusted operator configuration, never
from caller or model input, so knowledge paths cannot be steered from the
conversation. Markdown is split into plain text blocks; it is never
executed nor interpreted as system instructions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from receptionist.boundaries import (
    KnowledgeChunk,
    KnowledgeError,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeRetriever,
    KnowledgeService,
    KnowledgeSource,
    KnowledgeSourceDeclaration,
    KnowledgeSourceKind,
)


#: Frequent Spanish function words excluded from matching so a query
#: like "reparación de naves" does not match every chunk containing "de".
#: A fixed deterministic list, not an NLP pipeline.
STOPWORDS = frozenset(
    "de la el en y a los las del se con por para una uno unas unos "
    "que es son fue era cual cuál qué como cómo cuando donde quien "
    "mi mis tu tus su sus este esta estos estas ese esa eso aquel".split(" ")
)


def normalize_tokens(text: str) -> list[str]:
    """Lowercase, accent-insensitive, punctuation-free word tokens.

    Spanish-friendly: ``atención`` and ``atencion`` produce the same token.
    Stopwords are dropped so glue words never force a match.
    Deterministic: same input always yields the same token list.
    """
    folded = "".join(
        char
        for char in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(char)
    )
    cleaned = re.sub(r"[^a-z0-9ñü]+", " ", folded)
    return [token for token in cleaned.split(" ") if token and token not in STOPWORDS]


def _token_match(query_token: str, chunk_token: str) -> bool:
    """Exact match, or a prefix match for tokens long enough to be
    meaningful (covers reasonable partial keywords like ``horar``)."""
    if query_token == chunk_token:
        return True
    if len(query_token) >= 4 and len(chunk_token) >= 4:
        return chunk_token.startswith(query_token) or query_token.startswith(chunk_token)
    return False


class KeywordRetriever:
    """Local keyword retrieval over source chunks. No embeddings, no
    network, no external service. Ranking is deterministic: by matched
    token count, then source id, then chunk id."""

    def retrieve(
        self, query: str, chunks: list[KnowledgeChunk], limit: int
    ) -> list[KnowledgeChunk]:
        query_tokens = normalize_tokens(query)
        if not query_tokens:
            return []
        scored: list[tuple[int, str, str, KnowledgeChunk]] = []
        for chunk in chunks:
            # Match title plus search terms; search_text holds match-only
            # metadata (e.g. FAQ keywords) that must never be served as
            # content, and falls back to text when a source sets none.
            haystack = f"{chunk.title} {chunk.search_text or chunk.text}"
            chunk_tokens = normalize_tokens(haystack)
            score = sum(
                1
                for query_token in query_tokens
                if any(_token_match(query_token, token) for token in chunk_tokens)
            )
            if score > 0:
                scored.append((score, chunk.source_id, chunk.chunk_id, chunk))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [chunk for _, _, _, chunk in scored[: max(limit, 0)]]


@dataclass
class LocalKnowledgeService:
    """Combines configured sources through one retriever. The caller never
    learns which source produced a chunk; duplicates by (source, chunk)
    identity are served once. Any source or retriever failure becomes a
    FAILURE result; an empty match is an explicit NO_RESULT.

    Source ids must be unique: two sources sharing one id would produce
    indistinguishable chunk identities, so construction fails explicitly
    instead of silently dropping one source's content.
    """

    sources: list[KnowledgeSource]
    retriever: KnowledgeRetriever
    max_chunks: int = 3

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for source in self.sources:
            if source.source_id in seen:
                raise KnowledgeError(
                    f"duplicate knowledge source id: {source.source_id}"
                )
            seen.add(source.source_id)

    def query(self, query: KnowledgeQuery) -> KnowledgeResult:
        if not query.text.strip():
            return KnowledgeResult.no_result()
        try:
            chunks: list[KnowledgeChunk] = []
            for source in self.sources:
                chunks.extend(source.chunks())
        except KnowledgeError as error:
            return KnowledgeResult.failure(str(error))
        except Exception as error:  # fail closed behind the boundary
            return KnowledgeResult.failure(f"knowledge source failed: {error}")
        seen: set[tuple[str, str]] = set()
        unique: list[KnowledgeChunk] = []
        for chunk in chunks:
            key = (chunk.source_id, chunk.chunk_id)
            if key not in seen:
                seen.add(key)
                unique.append(chunk)
        try:
            ranked = self.retriever.retrieve(query.text, unique, self.max_chunks)
        except KnowledgeError as error:
            return KnowledgeResult.failure(str(error))
        except Exception as error:  # fail closed behind the boundary
            return KnowledgeResult.failure(f"knowledge retriever failed: {error}")
        if not ranked:
            return KnowledgeResult.no_result()
        return KnowledgeResult.found(tuple(ranked))


class FileKnowledgeSource:
    """One small local Markdown/text file, split into deterministic blocks.

    Chunking splits on blank lines; chunk ``i`` always carries the same id
    (``{source_id}#{i:04d}``) and provenance (``{path}#{i}``) for the same
    file content. Missing or undecodable files raise KnowledgeError when
    read; an empty file simply serves zero chunks. ``path`` must come from
    trusted operator configuration."""

    def __init__(self, source_id: str, path: str) -> None:
        self._source_id = source_id
        self._path = path

    @property
    def source_id(self) -> str:
        return self._source_id

    def chunks(self) -> list[KnowledgeChunk]:
        try:
            with open(self._path, encoding="utf-8") as handle:
                content = handle.read()
        except OSError as error:
            raise KnowledgeError(f"knowledge file unreadable: {self._path}: {error}") from error
        except UnicodeDecodeError as error:
            raise KnowledgeError(f"knowledge file malformed: {self._path}: {error}") from error
        blocks = [block.strip() for block in re.split(r"\n\s*\n", content)]
        result = []
        for index, block in enumerate(block for block in blocks if block):
            result.append(
                KnowledgeChunk(
                    source_id=self._source_id,
                    chunk_id=f"{self._source_id}#{index:04d}",
                    text=block,
                    origin=f"{self._path}#{index}",
                )
            )
        return result


def assemble_knowledge_sources(
    declarations: list[KnowledgeSourceDeclaration],
) -> list[KnowledgeSource]:
    """Build live sources from trusted operator declarations only.

    Disabled declarations are never built. The query/caller has no input
    here: every locator comes from configuration, so conversation text can
    never steer which files or databases are opened. Duplicate or blank
    source ids, unknown kinds, and missing locators fail explicitly with
    KnowledgeError instead of serving silently incomplete knowledge.
    """
    import sqlite3

    from receptionist.sqlite_storage import SQLiteFaqSource

    sources: list[KnowledgeSource] = []
    seen: set[str] = set()
    for declaration in declarations:
        if not declaration.enabled:
            continue
        if not declaration.source_id or not declaration.source_id.strip():
            raise KnowledgeError("knowledge source without id")
        if declaration.source_id in seen:
            raise KnowledgeError(
                f"duplicate knowledge source id: {declaration.source_id}"
            )
        seen.add(declaration.source_id)
        if not declaration.locator:
            raise KnowledgeError(
                f"knowledge source without locator: {declaration.source_id}"
            )
        if declaration.kind is KnowledgeSourceKind.FILE:
            sources.append(
                FileKnowledgeSource(
                    source_id=declaration.source_id, path=declaration.locator
                )
            )
        elif declaration.kind is KnowledgeSourceKind.FAQ:
            sources.append(SQLiteFaqSource(sqlite3.connect(declaration.locator)))
        else:
            raise KnowledgeError(
                f"unknown knowledge source kind: {declaration.kind!r}"
            )
    return sources
