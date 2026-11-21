"""SQLite veritabanı şeması (v2), yabancı anahtar zorunluluğu ve güvenli şema göçü."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

CURRENT_SCHEMA_VERSION = 2

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    question TEXT NOT NULL,
    language TEXT NOT NULL CHECK(language IN ('tr', 'en')),
    depth TEXT NOT NULL CHECK(depth IN ('fast', 'standard')),
    mode TEXT NOT NULL CHECK(mode IN ('demo', 'live')),
    model_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'partial', 'failed')),
    plan_summary TEXT NOT NULL DEFAULT '',
    scope_notes TEXT NOT NULL DEFAULT '',
    research_purpose TEXT NOT NULL DEFAULT '',
    timeframe TEXT NOT NULL DEFAULT '',
    key_concepts_json TEXT NOT NULL DEFAULT '[]',
    model_calls INTEGER NOT NULL DEFAULT 0,
    search_calls INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    token_measurement_type TEXT NOT NULL DEFAULT 'unmeasured',
    estimated_cost_usd REAL,
    research_quality TEXT NOT NULL DEFAULT 'high',
    error_type TEXT,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_started_at ON research_runs(started_at DESC);

CREATE TABLE IF NOT EXISTS sub_questions (
    run_id TEXT NOT NULL,
    sq_id TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    scope_note TEXT NOT NULL DEFAULT '',
    search_intent TEXT NOT NULL DEFAULT '',
    required_evidence TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK(status IN ('answered', 'partially_answered', 'conflicting', 'unanswered')),
    findings_json TEXT NOT NULL DEFAULT '[]',
    remaining_gaps_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (run_id, sq_id),
    FOREIGN KEY (run_id) REFERENCES research_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS search_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    round_no INTEGER NOT NULL,
    query TEXT NOT NULL,
    result_count INTEGER NOT NULL DEFAULT 0,
    source_ids_json TEXT NOT NULL DEFAULT '[]',
    triggered_by_claim_id TEXT,
    trigger_reason TEXT,
    missing_info TEXT,
    error_message TEXT,
    FOREIGN KEY (run_id) REFERENCES research_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sources (
    run_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    url TEXT NOT NULL,
    normalized_url TEXT NOT NULL,
    title TEXT NOT NULL,
    publisher TEXT NOT NULL,
    author TEXT,
    published_date TEXT,
    retrieved_at TEXT NOT NULL,
    language TEXT NOT NULL,
    provider TEXT NOT NULL,
    is_mock INTEGER NOT NULL DEFAULT 0,
    evidence_level TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'trusted_secondary_analysis',
    retrieval_method TEXT NOT NULL DEFAULT 'search_snippet',
    retrieved_content TEXT NOT NULL DEFAULT '',
    relevance_score REAL NOT NULL DEFAULT 1.0,
    relevance_assessment TEXT NOT NULL DEFAULT '',
    credibility_score REAL NOT NULL DEFAULT 0.7,
    is_accessible INTEGER NOT NULL DEFAULT 1,
    content_fingerprint TEXT NOT NULL DEFAULT '',
    duplicate_of_source_id TEXT,
    PRIMARY KEY (run_id, source_id),
    UNIQUE (run_id, normalized_url),
    FOREIGN KEY (run_id) REFERENCES research_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, duplicate_of_source_id) REFERENCES sources(run_id, source_id)
);

CREATE TABLE IF NOT EXISTS evidence_chunks (
    run_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    round_no INTEGER NOT NULL,
    excerpt_text TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    is_truncated INTEGER NOT NULL DEFAULT 0,
    query_used TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, chunk_id),
    FOREIGN KEY (run_id, source_id) REFERENCES sources(run_id, source_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claims (
    run_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    sq_id TEXT NOT NULL,
    statement TEXT NOT NULL,
    initial_source_id TEXT NOT NULL,
    initial_chunk_id TEXT NOT NULL,
    initial_quote TEXT NOT NULL,
    initial_quote_language TEXT NOT NULL DEFAULT 'tr',
    explanatory_note TEXT,
    round_created INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (run_id, claim_id),
    FOREIGN KEY (run_id, sq_id) REFERENCES sub_questions(run_id, sq_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, initial_source_id) REFERENCES sources(run_id, source_id),
    FOREIGN KEY (run_id, initial_chunk_id) REFERENCES evidence_chunks(run_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    round_no INTEGER NOT NULL,
    verdict TEXT NOT NULL CHECK(verdict IN ('supported', 'contradicted', 'conflicting', 'insufficient', 'irrelevant')),
    rationale TEXT NOT NULL,
    missing_info TEXT,
    follow_up_query TEXT,
    is_final INTEGER NOT NULL DEFAULT 1,
    causation_warning TEXT,
    guardrail_notes_json TEXT NOT NULL DEFAULT '[]',
    claim_nature TEXT NOT NULL DEFAULT 'direct_quote_measurement',
    numeric_context TEXT NOT NULL DEFAULT '',
    independent_support_count INTEGER NOT NULL DEFAULT 1,
    single_source_warning TEXT,
    UNIQUE (run_id, claim_id, round_no),
    FOREIGN KEY (run_id, claim_id) REFERENCES claims(run_id, claim_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS verification_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    verification_id INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    quote TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('supporting', 'contradicting', 'conflicting', 'context')),
    quote_language TEXT NOT NULL DEFAULT 'tr',
    explanatory_note TEXT,
    FOREIGN KEY (verification_id) REFERENCES verifications(id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, source_id) REFERENCES sources(run_id, source_id),
    FOREIGN KEY (run_id, chunk_id) REFERENCES evidence_chunks(run_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS reports (
    run_id TEXT PRIMARY KEY,
    language TEXT NOT NULL CHECK(language IN ('tr', 'en')),
    title TEXT NOT NULL,
    short_answer TEXT NOT NULL,
    report_markdown TEXT NOT NULL,
    report_json TEXT NOT NULL,
    used_source_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES research_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    node_name TEXT NOT NULL,
    round_no INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL,
    payload_summary TEXT NOT NULL DEFAULT '',
    UNIQUE (run_id, seq),
    FOREIGN KEY (run_id) REFERENCES research_runs(run_id) ON DELETE CASCADE
);
"""


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition};")


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Mevcut v1 veritabanını veri kaybetmeden v2 şemasına yükseltir."""
    _ensure_column(conn, "research_runs", "research_purpose", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "research_runs", "timeframe", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "research_runs", "key_concepts_json", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, "research_runs", "token_measurement_type", "TEXT NOT NULL DEFAULT 'unmeasured'")
    _ensure_column(conn, "research_runs", "estimated_cost_usd", "REAL")
    _ensure_column(conn, "research_runs", "research_quality", "TEXT NOT NULL DEFAULT 'high'")

    _ensure_column(conn, "sub_questions", "search_intent", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "sub_questions", "required_evidence", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "sub_questions", "findings_json", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, "sub_questions", "remaining_gaps_json", "TEXT NOT NULL DEFAULT '[]'")

    _ensure_column(conn, "sources", "author", "TEXT")
    _ensure_column(conn, "sources", "source_type", "TEXT NOT NULL DEFAULT 'trusted_secondary_analysis'")
    _ensure_column(conn, "sources", "retrieval_method", "TEXT NOT NULL DEFAULT 'search_snippet'")
    _ensure_column(conn, "sources", "retrieved_content", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "sources", "relevance_score", "REAL NOT NULL DEFAULT 1.0")
    _ensure_column(conn, "sources", "relevance_assessment", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "sources", "credibility_score", "REAL NOT NULL DEFAULT 0.7")
    _ensure_column(conn, "sources", "is_accessible", "INTEGER NOT NULL DEFAULT 1")

    # verifications CHECK kısıtını 'irrelevant' destekleyecek şekilde güvenli yeniden kur
    tbl_sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='verifications'"
    ).fetchone()
    tbl_sql = tbl_sql_row["sql"] if tbl_sql_row else ""
    if "irrelevant" not in tbl_sql:
        conn.execute("PRAGMA foreign_keys = OFF;")
        conn.executescript(
            """
            CREATE TABLE verifications_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                round_no INTEGER NOT NULL,
                verdict TEXT NOT NULL CHECK(verdict IN ('supported', 'contradicted', 'conflicting', 'insufficient', 'irrelevant')),
                rationale TEXT NOT NULL,
                missing_info TEXT,
                follow_up_query TEXT,
                is_final INTEGER NOT NULL DEFAULT 1,
                causation_warning TEXT,
                guardrail_notes_json TEXT NOT NULL DEFAULT '[]',
                claim_nature TEXT NOT NULL DEFAULT 'direct_quote_measurement',
                numeric_context TEXT NOT NULL DEFAULT '',
                independent_support_count INTEGER NOT NULL DEFAULT 1,
                single_source_warning TEXT,
                UNIQUE (run_id, claim_id, round_no),
                FOREIGN KEY (run_id, claim_id) REFERENCES claims(run_id, claim_id) ON DELETE CASCADE
            );
            INSERT INTO verifications_v2 (
                id, run_id, claim_id, round_no, verdict, rationale,
                missing_info, follow_up_query, is_final, causation_warning, guardrail_notes_json
            )
            SELECT
                id, run_id, claim_id, round_no, verdict, rationale,
                missing_info, follow_up_query, is_final, causation_warning, guardrail_notes_json
            FROM verifications;
            DROP TABLE verifications;
            ALTER TABLE verifications_v2 RENAME TO verifications;
            """
        )
        conn.execute("PRAGMA foreign_keys = ON;")


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Her bağlantıda yabancı anahtarları etkinleştirerek SQLite bağlantısı açar."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def init_db(db_path: Path) -> None:
    """Veritabanı tablolarını oluşturur ve gerekirse v2 şemasına göç uygular."""
    with closing(get_connection(db_path)) as conn:
        with conn:
            conn.executescript(SCHEMA_DDL)
            _migrate_v1_to_v2(conn)
            conn.execute("PRAGMA foreign_keys = ON;")
            existing = conn.execute(
                "SELECT version FROM schema_migrations WHERE version = ?",
                (CURRENT_SCHEMA_VERSION,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?)",
                    (
                        CURRENT_SCHEMA_VERSION,
                        datetime.now(UTC).isoformat(),
                        "P04 v2 şeması: kaynak kalitesi, irrelevant kararı, alt soru detayları ve kalite geçidi alanları",
                    ),
                )


def verify_migration_on_copy(db_path: Path) -> bool:
    """Mevcut veritabanının geçici kopyası üzerinde şema ve bütünlük denetimi yapar (Bölüm 7)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_path = Path(tmp_dir) / "migration_test_copy.db"
        if db_path.exists():
            shutil.copy2(db_path, copy_path)
        init_db(copy_path)
        with closing(get_connection(copy_path)) as conn:
            fk_row = conn.execute("PRAGMA foreign_keys;").fetchone()
            integrity = conn.execute("PRAGMA integrity_check;").fetchone()
            version_row = conn.execute(
                "SELECT MAX(version) AS v FROM schema_migrations"
            ).fetchone()
            return (
                fk_row is not None
                and fk_row[0] == 1
                and integrity is not None
                and integrity[0] == "ok"
                and version_row is not None
                and version_row["v"] == CURRENT_SCHEMA_VERSION
            )
