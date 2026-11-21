"""Canlı OpenAI ve Tavily adaptörlerinin hata ayrımı, boş arama ve tek tekrar testleri."""

from __future__ import annotations

import httpx
import pytest

from p04_research.config import AppConfig
from p04_research.models import PlannerOutput
from p04_research.providers.base import ProviderError
from p04_research.providers.openai_llm import OpenAILLMProvider
from p04_research.providers.tavily_search import TavilySearchProvider


def test_tavily_missing_key_raises_auth_error():
    cfg = AppConfig(tavily_api_key=None)
    prov = TavilySearchProvider(cfg)
    with pytest.raises(ProviderError) as exc_info:
        prov.search("test query", max_results=5, depth="fast")
    assert exc_info.value.error_type == "auth_error"


def test_tavily_empty_200_vs_http_errors():
    cfg = AppConfig(tavily_api_key="tvly-test-key-12345")

    # Boş HTTP 200 -> teknik arıza değil, 0 sonuç
    transport_empty = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"results": []})
    )
    prov_empty = TavilySearchProvider(cfg, http_client=httpx.Client(transport=transport_empty))
    resp = prov_empty.search("olmayan konu", max_results=5, depth="fast")
    assert resp.items == []
    assert resp.search_calls == 1

    # HTTP 401 -> auth_error
    transport_401 = httpx.MockTransport(lambda req: httpx.Response(401, json={"error": "bad key"}))
    prov_401 = TavilySearchProvider(cfg, http_client=httpx.Client(transport=transport_401))
    with pytest.raises(ProviderError) as exc_401:
        prov_401.search("q", max_results=5, depth="fast")
    assert exc_401.value.error_type == "auth_error"

    # HTTP 503 -> 1 tekrar denemeden sonra 200 alırsa kurtarır
    calls = {"n": 0}

    def handler_503_then_200(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "temporary"})
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.edu.tr/rapor",
                        "title": "Üniversite Raporu",
                        "content": "Bekleme süresi 6 dakika ölçüldü.",
                    }
                ]
            },
        )

    prov_retry = TavilySearchProvider(
        cfg, http_client=httpx.Client(transport=httpx.MockTransport(handler_503_then_200))
    )
    resp2 = prov_retry.search("bekleme suresi", max_results=5, depth="standard")
    assert calls["n"] == 2
    assert len(resp2.items) == 1
    assert resp2.items[0].published_date is None


def test_openai_missing_key_raises_auth_error():
    cfg = AppConfig(openai_api_key=None)
    prov = OpenAILLMProvider(cfg)
    with pytest.raises(ProviderError) as exc_info:
        prov.plan("Kampüs kütüphanesi sorusu?", "tr", "fast", 1)
    assert exc_info.value.error_type == "auth_error"


def test_openai_transient_timeout_retries_once():
    import openai

    cfg = AppConfig(openai_api_key="sk-test1234567890")
    attempts = {"n": 0}

    class FakeResponses:
        def parse(self, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))

            class FakeUsage:
                input_tokens = 120
                output_tokens = 45
                total_tokens = 165

            class FakeResp:
                output_parsed = PlannerOutput(
                    plan_summary="Test planı",
                    scope_notes="Test",
                    sub_questions=[
                        {"sq_id": "SQ1", "text": "Alt soru 1?", "scope_note": ""},
                        {"sq_id": "SQ2", "text": "Alt soru 2?", "scope_note": ""},
                    ],
                    initial_queries=["test sorgusu"],
                )
                usage = FakeUsage()

            return FakeResp()

    class FakeClient:
        responses = FakeResponses()

    prov = OpenAILLMProvider(cfg, client=FakeClient())  # type: ignore[arg-type]
    out, usage = prov.plan("Test araştırma sorusu nedir?", "tr", "fast", 1)
    assert attempts["n"] == 2
    assert usage.model_calls == 2
    assert usage.total_tokens == 165
    assert len(out.sub_questions) == 2
