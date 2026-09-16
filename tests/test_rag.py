"""Tests for the RAG-artefact scanner."""

from __future__ import annotations

from llm_apk_scanner.scanners.rag import RagArtefactScanner


def test_detects_all_positive_rag_paths(positive_rag_paths) -> None:
    scanner = RagArtefactScanner()
    detected = 0
    for path in positive_rag_paths:
        findings = scanner.scan([(path, "")])
        if findings:
            detected += 1
    recall = detected / len(positive_rag_paths)
    assert recall >= 0.85, f"RAG path recall {recall:.2%}"


def test_detects_langchain_class_reference() -> None:
    scanner = RagArtefactScanner()
    findings = scanner.scan(
        [("classes.dex", "L com/example/Foo$RetrievalQAChain.class")]
    )
    assert findings
    assert findings[0].title == "LangChain VectorStore reference"


def test_detects_embedding_model_id() -> None:
    scanner = RagArtefactScanner()
    findings = scanner.scan(
        [("classes.dex", 'model = "text-embedding-3-small"')]
    )
    assert findings
    assert "Embedding" in findings[0].title


def test_pickle_finding_is_high_severity() -> None:
    scanner = RagArtefactScanner()
    findings = scanner.scan([("assets/embeddings.pkl", "")])
    assert findings
    assert findings[0].severity.value == "high"


class TestAmbiguousVectorStoreNames:
    """`Chroma` and `Pinecone` are ordinary words, not just products.

    Matching them bare made every app bundling a video codec look like it
    shipped a vector store: `Chroma` is pervasive in Android media code
    (chroma subsampling, ChromaFormat, chromaticity). In the pilot this
    produced the great majority of `rag_artefact` findings.
    """

    def _titles(self, text: str) -> set[str]:
        return {f.title for f in RagArtefactScanner().scan([("classes.dex", text)])}

    def test_bare_chroma_in_media_code_is_not_a_finding(self):
        assert not self._titles(
            "ChromaFormat chromaSubsampling setChromaSitingHorz Chroma chromaticity"
        )

    def test_bare_pinecone_is_not_a_finding(self):
        assert not self._titles("PineconeForestLevel pinecone_texture Pinecone")

    def test_qualified_chroma_client_is_a_finding(self):
        assert self._titles("import chromadb; client = ChromaClient()")

    def test_qualified_pinecone_client_is_a_finding(self):
        assert self._titles("PineconeClient(api_key=...)")

    def test_distinctive_names_still_match(self):
        for token in ("VectorStore", "RetrievalQA", "Weaviate", "FAISS", "PGVector"):
            assert self._titles(f"some {token} reference"), token

    def test_bundled_chroma_database_path_still_matches(self):
        """Path evidence is unambiguous and must survive the tightening."""
        findings = RagArtefactScanner().scan(
            [("assets/chroma.sqlite3", "assets/chroma.sqlite3")]
        )
        assert findings
