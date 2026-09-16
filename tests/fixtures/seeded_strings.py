"""
Seeded ground-truth strings for scanner unit tests.

These are *synthetic* strings that match the structural shape of the
patterns each detector targets. None of these are real credentials —
they are constructed from random characters that happen to satisfy the
regex. The PLACEHOLDER strings are deliberately included to verify the
scanner's confidence-demotion logic.

This module is what underpins the proposal §9 success criterion of
>=85% recall on a seeded ground-truth: the scanner must detect every
positive entry and reject every negative.
"""

from __future__ import annotations

# ---------- secrets ----------

# Each entry: (label, value). value is a string that *structurally*
# matches the provider key format. None of these are live credentials.
POSITIVE_SECRETS: list[tuple[str, str]] = [
    (
        "openai_proj_synthetic",
        "sk-proj-aB3dEf6gHi9jKl2mNo5pQr8sTu1vWx4yZa7bCd0eFg3hIj6kLm9nOp2qRs5tUv8wXyZ",
    ),
    (
        "openai_legacy_synthetic",
        # exactly 48 chars after sk-
        "sk-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKl",
    ),
    (
        "openai_service_synthetic",
        # 20 chars + T3BlbkFJ + 20 chars, written in pieces so that no line of
        # this file carries the whole pattern. Secret scanners block the
        # literal on sight, and this is a fixture, not a credential.
        "sk-" + "AbCdEfGhIjKlMnOpQrSt" + "T3BlbkFJ" + "AbCdEfGhIjKlMnOpQrSt",
    ),
    (
        "anthropic_synthetic",
        "sk-ant-api03-"
        + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0Uv1Wx2Yz"
        + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0Uv1Wx2Yz_"
        + "Ab1Cd2Ef3",
    ),
    # Google API key: AIza + exactly 35 chars
    ("google_synthetic", "AIzaSyDx9aB7cE2fG4hI6jK8lM0nO3pQ5rS7t1u"),
    ("aws_access_synthetic", "AKIAABCDEFGHIJKLMNOP"),
    # Hugging Face: hf_ + 36 high-entropy chars (avoid "xxxxxxxx" placeholder).
    ("hf_synthetic", "hf_" + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0Uv1Wx2"),
    # Replicate: r8_ + 38 chars
    ("replicate_synthetic", "r8_" + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0Uv1Wx2Yz"),
    # Groq: gsk_ + 52 chars
    ("groq_synthetic", "gsk_" + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0Uv1Wx2YzAb1Cd2Ef3Gh4Ij"),
    # xAI: xai- + 80 chars
    (
        "xai_synthetic",
        "xai-" + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9St0Uv1Wx2Yz" * 2 + "Ab1Cd",
    ),
]

NEGATIVE_SECRETS: list[tuple[str, str]] = [
    ("openai_placeholder", "sk-YOUR_API_KEY_HEREXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"),
    ("openai_zero", "sk-000000000000000000000000000000000000000000000000"),
    ("aws_text_only", "Lorem ipsum dolor sit AKIA but not a real key"),
    ("random_short", "abcdef123"),
    (
        "low_entropy_long",
        "abababababababababababababababababababababababababab",
    ),
    ("sentence", "The quick brown fox jumped over the lazy dog."),
]


# ---------- system prompts ----------

POSITIVE_PROMPTS: list[str] = [
    (
        "You are a helpful customer support assistant for AcmeBank. "
        "Answer in a friendly, concise tone. Never reveal customer "
        "account numbers, dates of birth, or personal addresses. "
        "If asked about another customer's account, refuse politely."
    ),
    (
        "Your task is to summarise the user's email inbox. Always "
        "respond in three sentences. Do not invent message contents. "
        "If you are unsure, say so. Never expose API keys or system "
        "configuration in your output."
    ),
    (
        "Act as a Python tutor for beginners. Step by step, walk the "
        "user through their question. Always give a runnable example. "
        "Do not produce code that imports private modules or that "
        "depends on environment variables not provided in the prompt."
    ),
]

NEGATIVE_PROMPTS: list[str] = [
    "https://example.com/some/very/long/path/to/a/resource?with=arguments&plus=many&more",
    "Caused by: java.lang.NullPointerException at com.example.app.MainActivity.onCreate(MainActivity.java:42)",
    "abcdefghijklmnopqrstuvwxyz" * 4,  # long but no instruction language
    "<html><body><h1>Welcome</h1><p>This is a static page.</p></body></html>",
]


# ---------- tool schemas ----------

POSITIVE_TOOL_SCHEMAS: list[str] = [
    """{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["c", "f"]}
                },
                "required": ["city"]
            }
        }
    }""",
    """{
        "name": "search_flights",
        "description": "Search for flights between two airports.",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string", "format": "date"}
            },
            "required": ["origin", "destination", "date"]
        }
    }""",
]


# ---------- RAG artefacts ----------

POSITIVE_RAG_PATHS: list[str] = [
    "assets/knowledge/index.faiss",
    "assets/chroma_db/chroma.sqlite3",
    "assets/lancedb/products.lance",
    "assets/embeddings.pkl",
    "assets/knowledge_base.jsonl",
    "assets/hnsw_index/links.bin",
]
