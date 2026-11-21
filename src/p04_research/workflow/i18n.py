"""Türkçe ve İngilizce arayüz, durum etiketleri ve rapor başlıkları."""

from __future__ import annotations

from p04_research.models import (
    EvidenceLevel,
    RunStatus,
    SourceType,
    SubQuestionStatus,
    VerdictStatus,
)

TEXTS: dict[str, dict[str, str]] = {
    "tr": {
        "unknown": "Bilinmiyor",
        "not_measured": "Ölçülmedi",
        "estimated_tag": "Tahmini",
        "mock_badge": "Kurmaca eğitim verisi",
        "mock_tag": "Kurmaca",
        "live_tag": "Canlı Web Kaynağı",
        "report_heading_short": "Yönetici Özeti / Kısa Yanıt",
        "report_heading_purpose": "Araştırma Amacı",
        "report_heading_method": "Yöntem ve Kapsam",
        "report_heading_plan": "Alt Araştırma Soruları",
        "report_heading_findings": "Temel Bulgular",
        "report_heading_quant": "Sayısal Veriler ve Karşılaştırmalar",
        "report_heading_conflicts": "Çelişen Kaynaklar",
        "report_heading_counter": "Karşıt Kanıtlar",
        "report_heading_limitations": "Sınırlılıklar ve Elenen İddialar",
        "report_heading_assessment": "Genel Değerlendirme",
        "report_heading_open": "Açık Sorular",
        "report_heading_bibliography": "Kaynakça",
        "no_supported_findings": "Bu çalışmada doğrulama kurallarını geçen destekli bulgu oluşmadı.",
        "no_open_questions": "Açık kalan alt soru bulunmamaktadır.",
        "empty_search_notice": "Arama servisi bu sorgu için eşleşen kayıt döndürmedi (teknik arıza değil).",
        "technical_error_prefix": "Teknik servis hatası nedeniyle çalışma tamamlanamadı",
        "mode_demo": "Çevrimdışı Demo",
        "mode_live": "Canlı Araştırma",
        "depth_fast": "Hızlı",
        "depth_standard": "Standart",
        "quality_high": "Yüksek (Kalite Geçidinden Geçti)",
        "quality_needs_improvement": "İyileştirme Gerekli",
        "quality_insufficient": "Yetersiz Kanıt",
    },
    "en": {
        "unknown": "Unknown",
        "not_measured": "Not measured",
        "estimated_tag": "Estimated",
        "mock_badge": "Mock educational data",
        "mock_tag": "Mock",
        "live_tag": "Live Web Source",
        "report_heading_short": "Executive Summary / Short Answer",
        "report_heading_purpose": "Research Purpose",
        "report_heading_method": "Methodology & Scope",
        "report_heading_plan": "Research Plan & Sub-Questions",
        "report_heading_findings": "Key Findings",
        "report_heading_quant": "Quantitative Data & Comparisons",
        "report_heading_conflicts": "Conflicting Sources",
        "report_heading_counter": "Counter-Evidence",
        "report_heading_limitations": "Limitations & Rejected Claims",
        "report_heading_assessment": "Overall Assessment",
        "report_heading_open": "Open Questions",
        "report_heading_bibliography": "Bibliography",
        "no_supported_findings": "No supported findings passed verification rules in this run.",
        "no_open_questions": "There are no remaining open sub-questions.",
        "empty_search_notice": "The search service returned 0 matching records for this query (not a technical error).",
        "technical_error_prefix": "Run could not complete due to a technical service error",
        "mode_demo": "Offline Demo",
        "mode_live": "Live Research",
        "depth_fast": "Fast",
        "depth_standard": "Standard",
        "quality_high": "High (Passed Quality Gate)",
        "quality_needs_improvement": "Needs Improvement",
        "quality_insufficient": "Insufficient Evidence",
    },
}

VERDICT_LABELS: dict[str, dict[VerdictStatus, str]] = {
    "tr": {
        VerdictStatus.SUPPORTED: "Destekleniyor",
        VerdictStatus.CONTRADICTED: "Kaynak karşı çıkıyor",
        VerdictStatus.CONFLICTING: "Kaynaklar çelişiyor",
        VerdictStatus.INSUFFICIENT: "Kanıt yetersiz",
        VerdictStatus.IRRELEVANT: "Konuyla ilgisiz",
    },
    "en": {
        VerdictStatus.SUPPORTED: "Supported",
        VerdictStatus.CONTRADICTED: "Contradicted",
        VerdictStatus.CONFLICTING: "Conflicting sources",
        VerdictStatus.INSUFFICIENT: "Insufficient evidence",
        VerdictStatus.IRRELEVANT: "Irrelevant",
    },
}

SUB_QUESTION_LABELS: dict[str, dict[SubQuestionStatus, str]] = {
    "tr": {
        SubQuestionStatus.ANSWERED: "Yanıtlandı",
        SubQuestionStatus.PARTIALLY_ANSWERED: "Kısmen yanıtlandı",
        SubQuestionStatus.CONFLICTING: "Çelişkili",
        SubQuestionStatus.UNANSWERED: "Yanıtsız",
    },
    "en": {
        SubQuestionStatus.ANSWERED: "Answered",
        SubQuestionStatus.PARTIALLY_ANSWERED: "Partially answered",
        SubQuestionStatus.CONFLICTING: "Conflicting",
        SubQuestionStatus.UNANSWERED: "Unanswered",
    },
}

RUN_STATUS_LABELS: dict[str, dict[RunStatus, str]] = {
    "tr": {
        RunStatus.RUNNING: "Çalışıyor",
        RunStatus.COMPLETED: "Başarılı",
        RunStatus.PARTIAL: "Kısmi Sonuç",
        RunStatus.FAILED: "Başarısız",
    },
    "en": {
        RunStatus.RUNNING: "Running",
        RunStatus.COMPLETED: "Completed",
        RunStatus.PARTIAL: "Partial",
        RunStatus.FAILED: "Failed",
    },
}

EVIDENCE_LEVEL_LABELS: dict[str, dict[EvidenceLevel, str]] = {
    "tr": {
        EvidenceLevel.PRIMARY_RECORD: "Doğrudan ölçüm / birincil kayıt",
        EvidenceLevel.OFFICIAL_DOC: "Resmî / akademik yayın",
        EvidenceLevel.SECONDARY_REPORT: "İkincil rapor / haber",
        EvidenceLevel.SNIPPET_ONLY: "Yalnızca arama özeti parçası",
    },
    "en": {
        EvidenceLevel.PRIMARY_RECORD: "Primary record / direct measurement",
        EvidenceLevel.OFFICIAL_DOC: "Official / academic document",
        EvidenceLevel.SECONDARY_REPORT: "Secondary report",
        EvidenceLevel.SNIPPET_ONLY: "Search snippet only",
    },
}

SOURCE_TYPE_LABELS: dict[str, dict[SourceType, str]] = {
    "tr": {
        SourceType.ACADEMIC_PEER_REVIEWED: "Akademik ve hakemli yayın",
        SourceType.OFFICIAL_INSTITUTION: "Resmî kurum yayını",
        SourceType.RESEARCH_ORG_REPORT: "Araştırma kuruluşu özgün raporu",
        SourceType.CORPORATE_PRIMARY_DATA: "Kurumsal birincil veri",
        SourceType.TRUSTED_SECONDARY_ANALYSIS: "Güvenilir ikincil analiz",
        SourceType.NEWS_BLOG_OPINION: "Haber, blog veya görüş içeriği",
        SourceType.VIDEO_SOCIAL_MEDIA: "Video veya sosyal medya içeriği",
    },
    "en": {
        SourceType.ACADEMIC_PEER_REVIEWED: "Academic / peer-reviewed publication",
        SourceType.OFFICIAL_INSTITUTION: "Official institution publication",
        SourceType.RESEARCH_ORG_REPORT: "Research organization report",
        SourceType.CORPORATE_PRIMARY_DATA: "Corporate primary data",
        SourceType.TRUSTED_SECONDARY_ANALYSIS: "Trusted secondary analysis",
        SourceType.NEWS_BLOG_OPINION: "News, blog, or opinion piece",
        SourceType.VIDEO_SOCIAL_MEDIA: "Video or social media content",
    },
}


def t(key: str, language: str = "tr") -> str:
    lang = "en" if language == "en" else "tr"
    return TEXTS[lang].get(key, key)


def verdict_label(verdict: VerdictStatus, language: str = "tr") -> str:
    lang = "en" if language == "en" else "tr"
    return VERDICT_LABELS[lang].get(verdict, verdict.value)


def sq_status_label(status: SubQuestionStatus, language: str = "tr") -> str:
    lang = "en" if language == "en" else "tr"
    return SUB_QUESTION_LABELS[lang][status]


def run_status_label(status: RunStatus, language: str = "tr") -> str:
    lang = "en" if language == "en" else "tr"
    return RUN_STATUS_LABELS[lang][status]


def evidence_level_label(level: EvidenceLevel, language: str = "tr") -> str:
    lang = "en" if language == "en" else "tr"
    return EVIDENCE_LEVEL_LABELS[lang][level]


def source_type_label(source_type: SourceType, language: str = "tr") -> str:
    lang = "en" if language == "en" else "tr"
    return SOURCE_TYPE_LABELS[lang][source_type]


def quality_label(quality: str, language: str = "tr") -> str:
    mapping = {
        "high": t("quality_high", language),
        "needs_improvement": t("quality_needs_improvement", language),
        "insufficient_evidence": t("quality_insufficient", language),
    }
    return mapping.get(quality, quality)
