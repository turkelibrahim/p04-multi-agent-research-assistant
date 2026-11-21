"""Tavily Web Arama API canlı adaptörü (boş sonuç ile teknik hata ayrımı + 1 tekrar)."""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx

from p04_research.config import AppConfig
from p04_research.models import EvidenceLevel
from p04_research.providers.base import ProviderError, RawSearchItem, SearchResponse


def _infer_publisher(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or "Bilinmiyor"


def _infer_evidence_level(url: str, raw_content: str | None) -> EvidenceLevel:
    host = urlsplit(url).netloc.lower()
    if any(host.endswith(suffix) for suffix in (".gov", ".gov.tr", ".edu", ".edu.tr", ".ac.uk")):
        return EvidenceLevel.OFFICIAL_DOC
    if raw_content and len(raw_content.strip()) > 300:
        return EvidenceLevel.SECONDARY_REPORT
    return EvidenceLevel.SNIPPET_ONLY


class TavilySearchProvider:
    """Tavily Search REST API istemcisi (zaman aşımı ve hata ayrımı ile)."""

    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, config: AppConfig, http_client: httpx.Client | None = None) -> None:
        self.config = config
        self._client = http_client

    def search(self, query: str, max_results: int, depth: str) -> SearchResponse:
        if not self.config.has_tavily_key():
            raise ProviderError(
                "auth_error",
                "TAVILY_API_KEY tanımlı değil. Canlı web araması için .env dosyasına geçerli anahtar ekleyin.",
                retryable=False,
            )

        payload = {
            "api_key": self.config.tavily_api_key,
            "query": query,
            "search_depth": "advanced" if depth == "standard" else "basic",
            "max_results": max_results,
            "include_raw_content": False,
            "include_answer": False,
        }

        attempts = 0
        max_attempts = 2
        while attempts < max_attempts:
            attempts += 1
            try:
                if self._client is not None:
                    resp = self._client.post(
                        self.ENDPOINT,
                        json=payload,
                        timeout=float(self.config.search_timeout_sec),
                    )
                else:
                    with httpx.Client(timeout=float(self.config.search_timeout_sec)) as client:
                        resp = client.post(self.ENDPOINT, json=payload)

                if resp.status_code in (401, 403):
                    raise ProviderError(
                        "auth_error",
                        f"Tavily API anahtarı geçersiz veya yetkisiz (HTTP {resp.status_code}).",
                        retryable=False,
                    )
                if resp.status_code == 429:
                    raise ProviderError(
                        "quota_error",
                        "Tavily arama kotası veya hız sınırı aşıldı (HTTP 429).",
                        retryable=False,
                    )
                if resp.status_code >= 500:
                    if attempts < max_attempts:
                        continue
                    raise ProviderError(
                        "network_error",
                        f"Tavily sunucusu geçici hata döndürdü (HTTP {resp.status_code}).",
                        retryable=True,
                    )
                if resp.status_code >= 400:
                    raise ProviderError(
                        "api_error",
                        f"Tavily isteği reddedildi (HTTP {resp.status_code}).",
                        retryable=False,
                    )

                data = resp.json()
                raw_results = data.get("results") or []
                items: list[RawSearchItem] = []
                for entry in raw_results[:max_results]:
                    url = str(entry.get("url") or "").strip()
                    raw_full = entry.get("raw_content")
                    snippet = str(entry.get("content") or "").strip()
                    content = str(raw_full or snippet).strip()
                    title = str(entry.get("title") or url or "Başlıksız Kaynak").strip()
                    if not url:
                        continue
                    pub_date = entry.get("published_date")
                    pub_date_str = str(pub_date).strip() if pub_date else None
                    author_val = entry.get("author")
                    author_str = str(author_val).strip() if author_val else None
                    ret_method = "full_text" if (raw_full and len(str(raw_full).strip()) > len(snippet)) else "search_snippet"
                    accessible = bool(content)
                    items.append(
                        RawSearchItem(
                            url=url,
                            title=title,
                            publisher=_infer_publisher(url),
                            content=content or "",
                            published_date=pub_date_str,
                            language="en" if content.isascii() else "tr",
                            provider="tavily",
                            is_mock=False,
                            evidence_level=_infer_evidence_level(url, content),
                            author=author_str,
                            retrieval_method=ret_method if accessible else "inaccessible",
                            is_accessible=accessible,
                        )
                    )
                # HTTP 200 ile boş sonuç gelmesi teknik arıza DEĞİLDİR (K06).
                return SearchResponse(query=query, items=items, search_calls=1)

            except httpx.TimeoutException as exc:
                if attempts < max_attempts:
                    continue
                raise ProviderError(
                    "timeout",
                    f"Tavily araması {self.config.search_timeout_sec} sn içinde yanıt vermedi.",
                    retryable=True,
                ) from exc
            except httpx.RequestError as exc:
                if attempts < max_attempts:
                    continue
                raise ProviderError(
                    "network_error",
                    f"Tavily servisine ağ hatası nedeniyle ulaşılamadı: {type(exc).__name__}",
                    retryable=True,
                ) from exc

        raise ProviderError("network_error", "Tavily araması tamamlanamadı.", retryable=True)
