"""Uygulama yapılandırması, girdi doğrulama ve gizli bilgi maskeleme."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

MIN_QUESTION_CHARS = 10
MAX_QUESTION_CHARS = 2000

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"tvly-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._-]{12,}"),
]


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_int(raw: str | None, default: int) -> int:
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


@dataclass(slots=True)
class AppConfig:
    """Uygulama çalışma ve bütçe sınırları (Bölüm 8)."""

    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str | None = None
    tavily_api_key: str | None = None
    max_model_calls: int = 10
    max_search_calls_fast: int = 1
    max_search_calls_standard: int = 4
    max_initial_queries_standard: int = 3
    max_results_per_query: int = 5
    max_source_chars: int = 3500
    model_timeout_sec: int = 60
    search_timeout_sec: int = 20
    total_run_timeout_sec: int = 300
    max_quality_retries: int = 1
    db_path: Path = field(
        default_factory=lambda: _default_project_root() / "data" / "research_assistant.db"
    )

    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.strip())

    def has_tavily_key(self) -> bool:
        return bool(self.tavily_api_key and self.tavily_api_key.strip())

    def max_search_calls_for_depth(self, depth: str) -> int:
        if depth == "fast":
            return self.max_search_calls_fast
        return self.max_search_calls_standard

    def known_secrets(self) -> tuple[str, ...]:
        secrets: list[str] = []
        if self.openai_api_key:
            secrets.append(self.openai_api_key)
        if self.tavily_api_key:
            secrets.append(self.tavily_api_key)
        return tuple(secrets)


def load_config(
    env_path: Path | None = None,
    base_dir: Path | None = None,
) -> AppConfig:
    """Ortam değişkenlerinden ve varsa .env dosyasından ayarları yükler."""
    root = base_dir or _default_project_root()
    resolved_env = env_path or (root / ".env")
    fallback_example = root / ".env.example"

    file_values: dict[str, str] = {}
    if fallback_example.exists():
        raw_ex = dotenv_values(fallback_example)
        file_values.update({k: v for k, v in raw_ex.items() if v is not None and v.strip()})
    if resolved_env.exists():
        raw_vals = dotenv_values(resolved_env)
        file_values.update({k: v for k, v in raw_vals.items() if v is not None and v.strip()})

    def get_val(key: str, default: str | None = None) -> str | None:
        env_val = os.environ.get(key)
        if env_val is not None and env_val.strip() != "":
            return env_val.strip()
        f_val = file_values.get(key)
        if f_val is not None and f_val.strip() != "":
            return f_val.strip()
        return default

    raw_db = get_val("P04_DB_PATH", "data/research_assistant.db") or "data/research_assistant.db"
    db_candidate = Path(raw_db)
    if not db_candidate.is_absolute():
        db_candidate = root / db_candidate

    return AppConfig(
        openai_api_key=get_val("OPENAI_API_KEY"),
        openai_model=get_val("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini",
        openai_base_url=get_val("OPENAI_BASE_URL"),
        tavily_api_key=get_val("TAVILY_API_KEY"),
        max_model_calls=_parse_int(get_val("MAX_MODEL_CALLS"), 10),
        max_search_calls_fast=_parse_int(get_val("MAX_SEARCH_CALLS_FAST"), 1),
        max_search_calls_standard=_parse_int(get_val("MAX_SEARCH_CALLS_STANDARD"), 4),
        max_initial_queries_standard=_parse_int(get_val("MAX_INITIAL_QUERIES_STANDARD"), 3),
        max_results_per_query=_parse_int(get_val("MAX_RESULTS_PER_QUERY"), 5),
        max_source_chars=_parse_int(get_val("MAX_SOURCE_CHARS"), 3500),
        model_timeout_sec=_parse_int(get_val("MODEL_TIMEOUT_SEC"), 60),
        search_timeout_sec=_parse_int(get_val("SEARCH_TIMEOUT_SEC"), 20),
        total_run_timeout_sec=_parse_int(get_val("TOTAL_RUN_TIMEOUT_SEC"), 300),
        max_quality_retries=_parse_int(get_val("MAX_QUALITY_RETRIES"), 1),
        db_path=db_candidate,
    )


def validate_question(raw_question: str, language: str = "tr") -> tuple[bool, str, str]:
    """Kullanıcı sorusunu kurallara göre doğrular (boş / <10 / >2000 karakter reddi)."""
    cleaned = " ".join((raw_question or "").strip().split())
    if not cleaned:
        msg = (
            "Lütfen araştırılacak bir soru girin (boş bırakılamaz)."
            if language == "tr"
            else "Please enter a research question (cannot be empty)."
        )
        return False, "", msg
    if len(cleaned) < MIN_QUESTION_CHARS:
        msg = (
            f"Soru en az {MIN_QUESTION_CHARS} karakter olmalıdır (girilen: {len(cleaned)})."
            if language == "tr"
            else f"Question must be at least {MIN_QUESTION_CHARS} characters (got {len(cleaned)})."
        )
        return False, cleaned, msg
    if len(cleaned) > MAX_QUESTION_CHARS:
        msg = (
            f"Soru en fazla {MAX_QUESTION_CHARS} karakter olabilir (girilen: {len(cleaned)})."
            if language == "tr"
            else f"Question must be at most {MAX_QUESTION_CHARS} characters (got {len(cleaned)})."
        )
        return False, cleaned, msg
    return True, cleaned, ""


def redact_secrets(text: str, secrets: Iterable[str] = ()) -> str:
    """Metin içindeki API anahtarlarını ve gizli belirteçleri maskeler."""
    if not text:
        return text
    masked = text
    for secret in secrets:
        s = (secret or "").strip()
        if len(s) >= 4:
            masked = masked.replace(s, "[GIZLI_ANAHTAR]")
    for idx, pattern in enumerate(_SECRET_PATTERNS):
        if idx == 2:
            masked = pattern.sub(r"\1[GIZLI_ANAHTAR]", masked)
        else:
            masked = pattern.sub("[GIZLI_ANAHTAR]", masked)
    return masked
