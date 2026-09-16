"""
RAG-artefact detector.

Identifies bundled retrieval-augmented-generation corpora and vector
stores shipped inside the APK. The presence of an on-device RAG
corpus is a security finding for several reasons:
 - The corpus may contain proprietary/confidential content.
 - It exposes the assistant's "knowledge" to local extraction and
   poisoning (knowledge-base poisoning attacks).
 - Embedded vector stores often leak the embedding model's
   identifying parameters.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from llm_apk_scanner.models import (
    Finding,
    FindingKind,
    Severity,
    SourceLocation,
)
from llm_apk_scanner.scanners.base import BaseScanner


@dataclass(frozen=True)
class RagArtefactRule:
    name: str
    pattern: re.Pattern[str]
    description: str
    severity: Severity
    confidence: float


# File-extension and content-magic rules.
# Patterns match against asset *paths* and asset *content* signatures.
RAG_FILE_PATTERNS: list[RagArtefactRule] = [
    RagArtefactRule(
        name="FAISS index",
        pattern=re.compile(r"\.faiss$|/faiss(?:_index)?(?:/|$)|\bIDMap2?\b"),
        description="FAISS vector-store index file shipped in app assets.",
        severity=Severity.MEDIUM,
        confidence=0.85,
    ),
    RagArtefactRule(
        name="Chroma collection",
        pattern=re.compile(r"\.chroma$|/chroma(?:_db)?(?:/|$)|chroma\.sqlite3"),
        description="ChromaDB vector store shipped in app assets.",
        severity=Severity.MEDIUM,
        confidence=0.85,
    ),
    RagArtefactRule(
        name="LanceDB table",
        pattern=re.compile(r"\.lance$|/lancedb/"),
        description="LanceDB vector-store dataset shipped in app assets.",
        severity=Severity.MEDIUM,
        confidence=0.85,
    ),
    RagArtefactRule(
        name="HNSW serialized index",
        pattern=re.compile(r"\.hnsw$|/hnsw(?:_index)?(?:/|$)"),
        description="HNSW approximate-nearest-neighbour index file.",
        severity=Severity.MEDIUM,
        confidence=0.80,
    ),
    RagArtefactRule(
        name="Pickled embedding store",
        pattern=re.compile(r"embeddings?\.pkl$|vectors?\.pkl$"),
        description="Pickled embedding store. Pickled deserialization "
        "is also a code-execution risk on the device.",
        severity=Severity.HIGH,
        confidence=0.75,
    ),
    RagArtefactRule(
        name="Embedding JSONL/Parquet",
        pattern=re.compile(
            r"embeddings?\.(jsonl|parquet|csv)$|knowledge.*\.jsonl$"
        ),
        description="Plain embedding corpus (JSONL/Parquet/CSV).",
        severity=Severity.LOW,
        confidence=0.70,
    ),
]


# Content-level hints inside text/binary blobs. These catch
# string-table evidence even when no asset file matches.
RAG_CONTENT_HINTS: list[RagArtefactRule] = [
    RagArtefactRule(
        name="LangChain VectorStore reference",
        # Distinctive identifiers only. `Chroma` and `Pinecone` are ordinary
        # English words: `Chroma` in particular is pervasive in Android media
        # code (chroma subsampling, ChromaFormat, chromaticity), and matching
        # it bare made every app bundling a video codec look like it shipped
        # a vector store. Qualified forms of both are matched by the rule
        # below, and the path-based Chroma rule still catches a real bundled
        # ChromaDB.
        pattern=re.compile(
            r"\b(VectorStore|RetrievalQA|RetrievalQAChain|"
            r"ConversationalRetrievalChain|FAISS|"
            r"PGVector|Weaviate|LlamaIndex|VectorStoreIndex)\b"
        ),
        description="LangChain/LlamaIndex retrieval primitive referenced "
        "in code.",
        severity=Severity.INFO,
        confidence=0.55,
    ),
    RagArtefactRule(
        name="Qualified vector-store client reference",
        # The qualified forms of the ambiguous names: these cannot occur by
        # accident in media or graphics code.
        pattern=re.compile(
            r"\b(chromadb|ChromaClient|Chroma\.from_|langchain_chroma|"
            r"PineconeClient|pinecone\.Index|pinecone-client|PineconeStore|"
            r"QdrantClient|MilvusClient|qdrant_client|pymilvus)\b"
        ),
        description="Vector-database client library referenced in code.",
        severity=Severity.INFO,
        confidence=0.7,
    ),
    RagArtefactRule(
        name="Embedding model identifier",
        pattern=re.compile(
            r"\b(text-embedding-(?:ada-002|3-small|3-large)|"
            r"voyage-(?:large-2|2)|"
            r"embed-english-v\d|embed-multilingual-v\d|"
            r"jina-embeddings-v\d|"
            r"all-MiniLM-L\d+-v\d|all-mpnet-base-v\d)\b"
        ),
        description="Embedding-model identifier hard-coded in app.",
        severity=Severity.INFO,
        confidence=0.85,
    ),
]


class RagArtefactScanner(BaseScanner):
    """Detects bundled vector stores and retrieval-augmented corpora."""

    name = "rag_artefacts"

    def __init__(
        self,
        file_rules: list[RagArtefactRule] | None = None,
        content_rules: list[RagArtefactRule] | None = None,
    ) -> None:
        self.file_rules = file_rules or RAG_FILE_PATTERNS
        self.content_rules = content_rules or RAG_CONTENT_HINTS

    def scan(self, sources: list[tuple[str, str]]) -> list[Finding]:
        findings: list[Finding] = []
        for source_file, content in sources:
            findings.extend(self._scan_file_path(source_file))
            findings.extend(self._scan_content(source_file, content))
        return findings

    def _scan_file_path(self, source_file: str) -> list[Finding]:
        findings: list[Finding] = []
        for rule in self.file_rules:
            if rule.pattern.search(source_file):
                findings.append(
                    Finding(
                        kind=FindingKind.RAG_ARTEFACT,
                        severity=rule.severity,
                        title=rule.name,
                        description=rule.description,
                        location=SourceLocation(
                            file_path=source_file,
                            container=os.path.dirname(source_file) or None,
                        ),
                        evidence=os.path.basename(source_file),
                        confidence=rule.confidence,
                        metadata={"rule": rule.name, "match": "path"},
                    )
                )
        return findings

    def _scan_content(
        self, source_file: str, content: str
    ) -> list[Finding]:
        findings: list[Finding] = []
        for rule in self.content_rules:
            for m in rule.pattern.finditer(content):
                findings.append(
                    Finding(
                        kind=FindingKind.RAG_ARTEFACT,
                        severity=rule.severity,
                        title=rule.name,
                        description=rule.description,
                        location=SourceLocation(
                            file_path=source_file,
                            line_or_offset=m.start(),
                        ),
                        evidence=m.group(0),
                        confidence=rule.confidence,
                        metadata={"rule": rule.name, "match": "content"},
                    )
                )
        return findings
