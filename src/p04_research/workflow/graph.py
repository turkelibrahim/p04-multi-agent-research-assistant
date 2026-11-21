"""LangGraph StateGraph akışı, kurallı yönlendirici, Quality Gate ve ana çalıştırma girişi."""

from __future__ import annotations

import sqlite3
import time
import uuid
from datetime import UTC, datetime

from langgraph.graph import END, START, StateGraph

from p04_research.config import AppConfig, validate_question
from p04_research.db.repository import ResearchRepository
from p04_research.models import (
    ResearchDepth,
    RunBundle,
    RunLanguage,
    RunMetadata,
    RunMode,
    RunStatus,
)
from p04_research.providers.base import LLMProvider, SearchProvider
from p04_research.providers.live_extractive import ResilientLiveLLMProvider
from p04_research.providers.mock_demo import MockDemoProvider
from p04_research.providers.tavily_search import TavilySearchProvider
from p04_research.workflow.nodes import EventCallback, _pick_followup_candidate, make_nodes
from p04_research.workflow.state import ResearchGraphState


def estimate_cost_usd(
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    """gpt-4.1-mini referans tarifesine göre ($0.40/1M giriş, $1.60/1M çıkış) tahmini maliyeti hesaplar."""
    if prompt_tokens is None or completion_tokens is None:
        return None
    cost = (prompt_tokens * 0.40 + completion_tokens * 1.60) / 1_000_000.0
    return round(cost, 6)


def build_research_graph(
    config: AppConfig,
    llm_provider: LLMProvider,
    search_provider: SearchProvider,
    on_event: EventCallback | None = None,
):
    """Beş araştırma düğümü, koşullu yönlendirici ve Quality Gate içeren LangGraph StateGraph nesnesini derler."""
    nodes = make_nodes(config, llm_provider, search_provider, on_event=on_event)

    def route_after_planner(state: ResearchGraphState) -> str:
        if state["fatal_error_message"] or not state["pending_queries"]:
            return "reporter"
        return "researcher"

    def route_after_researcher(state: ResearchGraphState) -> str:
        if state["fatal_error_message"] and len(state["chunks"]) == 0:
            return "reporter"
        if len(state["chunks"]) == 0:
            return "reporter"
        return "analyst"

    def route_after_analyst(state: ResearchGraphState) -> str:
        if state["fatal_error_message"]:
            return "reporter"
        return "verifier"

    def route_after_verifier(state: ResearchGraphState) -> str:
        if state["fatal_error_message"]:
            return "reporter"
        if state["depth"] != "standard":
            return "reporter"
        if state["round_no"] >= 2:
            return "reporter"
        if state["search_calls_used"] >= config.max_search_calls_standard:
            return "reporter"
        if state["model_calls_used"] + 3 > config.max_model_calls:
            return "reporter"
        if (time.monotonic() - state["started_monotonic"]) >= max(1, config.total_run_timeout_sec - 5):
            return "reporter"
        if _pick_followup_candidate(state) is not None:
            return "plan_followup"
        return "reporter"

    def route_after_quality_gate(state: ResearchGraphState) -> str:
        if state["status"] == RunStatus.FAILED.value:
            return END
        if not state.get("quality_passed", True) and state.get("quality_feedback"):
            if state.get("quality_retry_count", 0) <= config.max_quality_retries:
                # Yalnızca quality_gate_node retry_count'u artırıp tekrar istediğinde döner
                last_ev = state["events"][-1] if state["events"] else None
                if last_ev and last_ev.node_name == "quality_gate" and last_ev.status == "decision":
                    return "reporter"
        return END

    graph = StateGraph(ResearchGraphState)
    graph.add_node("planner", nodes["planner"])
    graph.add_node("researcher", nodes["researcher"])
    graph.add_node("analyst", nodes["analyst"])
    graph.add_node("verifier", nodes["verifier"])
    graph.add_node("plan_followup", nodes["plan_followup"])
    graph.add_node("reporter", nodes["reporter"])
    graph.add_node("quality_gate", nodes["quality_gate"])

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"researcher": "researcher", "reporter": "reporter"},
    )
    graph.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {"analyst": "analyst", "reporter": "reporter"},
    )
    graph.add_conditional_edges(
        "analyst",
        route_after_analyst,
        {"verifier": "verifier", "reporter": "reporter"},
    )
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"plan_followup": "plan_followup", "reporter": "reporter"},
    )
    graph.add_edge("plan_followup", "researcher")
    graph.add_edge("reporter", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {"reporter": "reporter", END: END},
    )

    return graph.compile()


def execute_research_run(
    question: str,
    language: str = "tr",
    depth: str = "standard",
    mode: str = "demo",
    config: AppConfig | None = None,
    repository: ResearchRepository | None = None,
    idempotency_key: str | None = None,
    llm_provider: LLMProvider | None = None,
    search_provider: SearchProvider | None = None,
    on_event: EventCallback | None = None,
) -> tuple[RunBundle, bool, str | None]:
    """Soruyu doğrular, LangGraph akışını çalıştırır ve sonucu isteğe bağlı SQLite'a kaydeder."""
    cfg = config or AppConfig()
    lang = "en" if language == "en" else "tr"
    depth_val = "fast" if depth == "fast" else "standard"
    mode_val = "live" if mode == "live" else "demo"

    ok, cleaned_q, err_msg = validate_question(question, lang)
    if not ok:
        raise ValueError(err_msg)

    idem = idempotency_key or uuid.uuid4().hex
    if repository is not None:
        try:
            existing_id = repository.get_run_id_by_idempotency_key(idem)
            if existing_id is not None:
                loaded = repository.load_run_bundle(existing_id)
                if loaded is not None:
                    return loaded, True, None
        except (sqlite3.Error, OSError):
            pass

    if llm_provider is None or search_provider is None:
        if mode_val == "demo":
            demo = MockDemoProvider()
            llm_provider = llm_provider or demo
            search_provider = search_provider or demo
        else:
            llm_provider = llm_provider or ResilientLiveLLMProvider(cfg)
            search_provider = search_provider or TavilySearchProvider(cfg)

    model_label = f"{cfg.openai_model} (Demo)" if mode_val == "demo" else cfg.openai_model
    run_id = f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    t_mono = time.monotonic()

    initial_state: ResearchGraphState = {
        "run_id": run_id,
        "idempotency_key": idem,
        "question": cleaned_q,
        "language": lang,
        "depth": depth_val,
        "mode": mode_val,
        "model_name": model_label,
        "started_at": started_at,
        "started_monotonic": t_mono,
        "round_no": 1,
        "plan_summary": "",
        "scope_notes": "",
        "research_purpose": "",
        "timeframe": "",
        "key_concepts": [],
        "sub_questions": [],
        "pending_queries": [],
        "executed_queries": [],
        "follow_up_reason": None,
        "follow_up_claim_id": None,
        "follow_up_missing_info": None,
        "search_rounds": [],
        "sources": [],
        "chunks": [],
        "claims": [],
        "verifications": [],
        "events": [],
        "model_calls_used": 0,
        "search_calls_used": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "token_measurement_type": "unmeasured",
        "fatal_error_type": None,
        "fatal_error_message": None,
        "technical_search_failure": False,
        "status": RunStatus.RUNNING.value,
        "quality_retry_count": 0,
        "quality_feedback": [],
        "quality_passed": True,
        "research_quality": "high",
        "report": None,
    }

    compiled = build_research_graph(cfg, llm_provider, search_provider, on_event=on_event)
    final_state = compiled.invoke(initial_state)

    finished_at = datetime.now(UTC).isoformat(timespec="seconds")
    dur_ms = int((time.monotonic() - t_mono) * 1000)
    final_status = RunStatus(final_state["status"]) if final_state.get("status") else RunStatus.FAILED
    est_cost = estimate_cost_usd(final_state["prompt_tokens"], final_state["completion_tokens"])

    metadata = RunMetadata(
        run_id=run_id,
        idempotency_key=idem,
        question=cleaned_q,
        language=RunLanguage(lang),
        depth=ResearchDepth(depth_val),
        mode=RunMode(mode_val),
        model_name=model_label,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=dur_ms,
        status=final_status,
        plan_summary=final_state["plan_summary"],
        scope_notes=final_state["scope_notes"],
        research_purpose=final_state.get("research_purpose", ""),
        timeframe=final_state.get("timeframe", ""),
        key_concepts=final_state.get("key_concepts", []),
        model_calls=final_state["model_calls_used"],
        search_calls=final_state["search_calls_used"],
        prompt_tokens=final_state["prompt_tokens"],
        completion_tokens=final_state["completion_tokens"],
        total_tokens=final_state["total_tokens"],
        token_measurement_type=final_state.get("token_measurement_type", "unmeasured"),
        estimated_cost_usd=est_cost,
        research_quality=final_state.get("research_quality", "high"),
        error_type=final_state["fatal_error_type"],
        error_message=final_state["fatal_error_message"],
    )

    bundle = RunBundle(
        metadata=metadata,
        sub_questions=final_state["sub_questions"],
        search_rounds=final_state["search_rounds"],
        sources=final_state["sources"],
        chunks=final_state["chunks"],
        claims=final_state["claims"],
        verifications=final_state["verifications"],
        events=final_state["events"],
        report=final_state["report"],
    )

    if repository is not None:
        persisted, persist_err = repository.save_run_bundle(bundle)
        return bundle, persisted, persist_err

    return bundle, False, None
