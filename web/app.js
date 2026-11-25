"use strict";

const state = {
  config: null,
  runs: [],
  activeBundle: null,
  activeTab: "report",
  form: {
    mode: "demo",
    depth: "standard",
    language: "tr",
  },
  isRunning: false,
};

const els = {
  sidebar: document.getElementById("sidebar"),
  backdrop: document.getElementById("sidebar-backdrop"),
  openSidebarBtn: document.getElementById("open-sidebar-btn"),
  closeSidebarBtn: document.getElementById("close-sidebar-btn"),
  brandHomeBtn: document.getElementById("brand-home-btn"),
  newResearchBtn: document.getElementById("new-research-btn"),
  mobileNewBtn: document.getElementById("mobile-new-btn"),
  historySearch: document.getElementById("history-search"),
  historyList: document.getElementById("history-list"),
  clearHistoryBtn: document.getElementById("clear-history-btn"),
  envStatusDot: document.getElementById("env-status-dot"),
  envStatusText: document.getElementById("env-status-text"),

  viewComposer: document.getElementById("view-composer"),
  viewRunning: document.getElementById("view-running"),
  viewReport: document.getElementById("view-report"),

  researchForm: document.getElementById("research-form"),
  questionInput: document.getElementById("question-input"),
  submitBtn: document.getElementById("submit-btn"),
  formError: document.getElementById("form-error"),
  modeHint: document.getElementById("mode-hint"),
  presetsGrid: document.getElementById("presets-grid"),

  runningQuestion: document.getElementById("running-question"),
  liveSteps: document.getElementById("live-steps"),

  reportTitle: document.getElementById("report-title"),
  reportMeta: document.getElementById("report-meta"),
  dlMdBtn: document.getElementById("dl-md-btn"),
  dlPdfBtn: document.getElementById("dl-pdf-btn"),
  reportNotice: document.getElementById("report-notice"),
  tabSourcesCount: document.getElementById("tab-sources-count"),
  headingSummary: document.getElementById("heading-summary"),
  headingFindings: document.getElementById("heading-findings"),
  headingConflicts: document.getElementById("heading-conflicts"),
  headingSources: document.getElementById("heading-sources"),
  reportSummary: document.getElementById("report-summary"),
  reportFindings: document.getElementById("report-findings"),
  sectionConflicts: document.getElementById("section-conflicts"),
  reportConflicts: document.getElementById("report-conflicts"),
  reportBibliography: document.getElementById("report-bibliography"),
  auditDetails: document.getElementById("audit-details"),
  auditBody: document.getElementById("audit-body"),

  sourceCards: document.getElementById("source-cards"),
  sectionProvenance: document.getElementById("section-provenance"),
  claimSelect: document.getElementById("claim-select"),
  provenanceDetail: document.getElementById("provenance-detail"),

  processStats: document.getElementById("process-stats"),
  processFollowups: document.getElementById("process-followups"),
  processTimeline: document.getElementById("process-timeline"),
};

function esc(str) {
  return String(str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatDateShort(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso).slice(0, 16).replace("T", " ");
  return d.toLocaleDateString("tr-TR", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function formatDateTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso).slice(0, 16).replace("T", " ");
  return d.toLocaleString("tr-TR", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderSourceAnchor(src) {
  if (!src) return "";
  const label = esc(src.display_label || src.clean_title || src.source_id);
  if (src.has_external_url && src.url) {
    return `<a class="source-link" href="${esc(src.url)}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  }
  return `<strong>${label}</strong>`;
}

function setView(viewName) {
  els.viewComposer.classList.toggle("hidden", viewName !== "composer");
  els.viewRunning.classList.toggle("hidden", viewName !== "running");
  els.viewReport.classList.toggle("hidden", viewName !== "report");
}

function setActiveTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    const active = btn.dataset.tab === tabName;
    btn.classList.toggle("is-active", active);
    btn.setAttribute("aria-selected", String(active));
  });
  document.getElementById("panel-report").classList.toggle("hidden", tabName !== "report");
  document.getElementById("panel-sources").classList.toggle("hidden", tabName !== "sources");
  document.getElementById("panel-process").classList.toggle("hidden", tabName !== "process");
}

function closeMobileSidebar() {
  els.sidebar.classList.remove("is-open");
  els.backdrop.classList.add("hidden");
}

function syncSegmentedControls() {
  document.querySelectorAll(".seg-group").forEach((group) => {
    const key = group.dataset.control;
    const current = state.form[key];
    group.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.value === current);
    });
  });

  if (state.form.mode === "live") {
    if (state.config?.has_tavily_key) {
      els.modeHint.textContent = "Canlı web araması aktif — güncel web kaynakları üzerinden doğrulama yapılır.";
    } else {
      els.modeHint.textContent = "Canlı arama için .env dosyasında Tavily API anahtarı tanımlı olmalıdır.";
    }
  } else {
    els.modeHint.textContent = "Demo modu aktif — internet bağlantısı olmadan örnek veriyle çalışır.";
  }
}

function renderPresets() {
  const presets = state.config?.presets || [];
  els.presetsGrid.innerHTML = presets
    .map(
      (p) =>
        `<button type="button" class="preset-chip" data-preset="${esc(p.key)}" title="${esc(p.description)}">${esc(p.label)}</button>`
    )
    .join("");

  els.presetsGrid.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const preset = presets.find((x) => x.key === chip.dataset.preset);
      if (!preset) return;
      els.questionInput.value =
        state.form.language === "en" ? preset.question_en : preset.question_tr;
      state.form.mode = "demo";
      syncSegmentedControls();
      els.formError.classList.add("hidden");
      els.questionInput.focus();
    });
  });
}

function renderHistory() {
  const filter = (els.historySearch.value || "").trim().toLowerCase();
  const filtered = state.runs.filter((r) => {
    if (!filter) return true;
    return (
      (r.display_title || "").toLowerCase().includes(filter) ||
      (r.question || "").toLowerCase().includes(filter)
    );
  });

  els.clearHistoryBtn.classList.toggle("hidden", state.runs.length === 0);

  if (filtered.length === 0) {
    els.historyList.innerHTML = `<div class="history-empty">${
      state.runs.length === 0 ? "Henüz kayıtlı araştırma yok." : "Eşleşen kayıt bulunamadı."
    }</div>`;
    return;
  }

  const activeId = state.activeBundle?.metadata?.run_id;
  els.historyList.innerHTML = filtered
    .map((r) => {
      const isActive = r.run_id === activeId && !els.viewReport.classList.contains("hidden");
      const modeLbl = r.mode === "demo" ? "Demo" : "Web";
      return `
        <div class="history-item ${isActive ? "is-active" : ""}" data-run-id="${esc(r.run_id)}" role="listitem">
          <div class="history-item-body">
            <div class="history-item-title">${esc(r.display_title || r.question)}</div>
            <div class="history-item-meta">${esc(formatDateShort(r.started_at))} · ${modeLbl} · ${r.supported_claim_count} bulgu</div>
          </div>
          <button type="button" class="history-del-btn" data-del-id="${esc(r.run_id)}" aria-label="Kaydı sil" title="Sil">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
          </button>
        </div>
      `;
    })
    .join("");

  els.historyList.querySelectorAll(".history-item").forEach((item) => {
    item.addEventListener("click", (ev) => {
      if (ev.target.closest(".history-del-btn")) return;
      loadRun(item.dataset.runId);
      closeMobileSidebar();
    });
  });

  els.historyList.querySelectorAll(".history-del-btn").forEach((btn) => {
    btn.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      await deleteRun(btn.dataset.delId);
    });
  });
}

function renderProvenanceDetail(claimId) {
  const list = state.activeBundle?.provenance || [];
  const p = list.find((item) => item.claim_id === claimId) || list[0];
  if (!p) {
    els.provenanceDetail.innerHTML = "";
    return;
  }
  const evHtml = (p.evidence_items || [])
    .map((e) => `<div class="quote-block">"${esc(e.quote)}"</div>`)
    .join("");

  els.provenanceDetail.innerHTML = `
    <div class="prov-box">
      <h4>1. Tur İlk Dayanak</h4>
      ${p.sub_question ? `<p class="finding-note" style="margin-top:0;margin-bottom:8px;">${esc(p.sub_question)}</p>` : ""}
      <div class="quote-block">"${esc(p.initial_quote)}"</div>
      ${
        p.round1_verdict_label
          ? `<p class="finding-note"><strong>${esc(p.round1_verdict_label)}:</strong> ${esc(p.round1_rationale || "")}</p>`
          : ""
      }
    </div>
    <div class="prov-box">
      <h4>Sonuç ve Doğrulama</h4>
      ${p.followup_query ? `<p class="finding-note" style="margin-top:0;">2. Tur sorgusu: ${esc(p.followup_query)}</p>` : ""}
      <p style="margin:6px 0;"><strong>${esc(p.final_verdict_label || "-")}</strong>${p.final_round ? ` (Tur ${p.final_round})` : ""}</p>
      <p class="finding-note" style="margin-top:0;">${esc(p.final_rationale || "")}</p>
      ${evHtml}
    </div>
  `;
}

function renderBundle(bundle) {
  state.activeBundle = bundle;
  const m = bundle.metadata;
  const rep = bundle.report;
  const isEn = m.language === "en";

  els.reportTitle.textContent = rep?.display_title || m.display_title || m.question;
  const findingCount = rep?.findings?.length ?? 0;
  const bibCount = rep?.bibliography?.length ?? 0;
  const demoPill =
    m.mode === "demo"
      ? `<span class="pill pill-demo">${isEn ? "Sample Data" : "Örnek Veri"}</span>`
      : "";
  els.reportMeta.innerHTML = `
    <span>${findingCount} ${isEn ? "findings" : "bulgu"}</span>
    <span>·</span>
    <span>${bibCount} ${isEn ? "sources" : "kaynak"}</span>
    <span>·</span>
    <span>${esc(formatDateTime(m.started_at))}</span>
    ${demoPill}
  `;

  els.dlMdBtn.href = `/api/runs/${encodeURIComponent(m.run_id)}/export/md`;
  els.dlPdfBtn.href = `/api/runs/${encodeURIComponent(m.run_id)}/export/pdf`;
  els.tabSourcesCount.textContent = String(bundle.sources?.length ?? 0);

  const noticeText = rep?.error_notice || bundle.persist_error || "";
  els.reportNotice.textContent = noticeText;
  els.reportNotice.classList.toggle("hidden", !noticeText);

  els.headingSummary.textContent = isEn ? "Short Answer" : "Kısa Yanıt";
  els.headingFindings.textContent = isEn ? "Key Findings" : "Temel Bulgular";
  els.headingConflicts.textContent = isEn ? "Conflicting Sources" : "Çelişen Kaynaklar";
  els.headingSources.textContent = isEn ? "Sources" : "Kaynaklar";

  // 1. Kısa Yanıt
  els.reportSummary.innerHTML = rep?.short_answer_html || esc(rep?.short_answer || "");

  // 2. Temel Bulgular
  const findings = rep?.findings || [];
  if (findings.length === 0) {
    els.reportFindings.innerHTML = `<div class="finding-card"><p class="finding-note" style="margin:0;">${
      isEn
        ? "No findings passed verification rules in this run."
        : "Bu çalışmada doğrulama kurallarını geçen destekli bulgu oluşmadı."
    }</p></div>`;
  } else {
    els.reportFindings.innerHTML = findings
      .map((f) => {
        const srcHtml = f.source
          ? `<div class="finding-source">${isEn ? "Source" : "Kaynak"}: ${renderSourceAnchor(f.source)}</div>`
          : "";
        if (f.is_duplicate_of_quote && f.primary_quote) {
          return `
            <article class="finding-card">
              <p class="finding-text">${esc(f.primary_quote)}</p>
              ${srcHtml}
            </article>
          `;
        }
        return `
          <article class="finding-card">
            <p class="finding-statement">${f.statement_html || esc(f.statement)}</p>
            ${f.primary_quote ? `<div class="quote-block">"${esc(f.primary_quote)}"</div>` : ""}
            ${f.explanatory_note ? `<p class="finding-note">${esc(f.explanatory_note)}</p>` : ""}
            ${f.causation_warning ? `<p class="finding-note">${esc(f.causation_warning)}</p>` : ""}
            ${srcHtml}
          </article>
        `;
      })
      .join("");
  }

  // 3. Çelişen Kaynaklar
  const conflicts = rep?.conflicts || [];
  els.sectionConflicts.classList.toggle("hidden", conflicts.length === 0);
  els.reportConflicts.innerHTML = conflicts
    .map(
      (c) => `
      <article class="conflict-card">
        <p class="finding-statement">${esc(c.statement)}</p>
        <p class="finding-note" style="margin-top:0;">${esc(c.explanation)}</p>
        <div class="conflict-grid">
          <div>
            <div class="quote-block">"${esc(c.side_a.quote)}"</div>
            <div class="finding-source">${renderSourceAnchor(c.side_a.source)}</div>
          </div>
          <div>
            <div class="quote-block">"${esc(c.side_b.quote)}"</div>
            <div class="finding-source">${renderSourceAnchor(c.side_b.source)}</div>
          </div>
        </div>
      </article>
    `
    )
    .join("");

  // 4. Kaynaklar (yalnızca temiz kaynak adı + gizli link; tarihsizse 'Bilinmiyor' gürültüsü yok)
  const bib = rep?.bibliography || [];
  if (bib.length === 0) {
    els.reportBibliography.innerHTML = `<li>${isEn ? "No sources recorded." : "Kaynak kaydedilmedi."}</li>`;
  } else {
    els.reportBibliography.innerHTML = bib
      .map((b) => {
        const datePart = b.published_date ? ` <span class="bib-date">· ${esc(b.published_date)}</span>` : "";
        return `<li>${renderSourceAnchor(b)}${datePart}</li>`;
      })
      .join("");
  }

  // 5. Denetim Detayları ve Alt Sorular
  const auditChunks = [];
  if ((m.question || "").trim() !== (rep?.display_title || "").trim()) {
    auditChunks.push(`
      <div>
        <div class="audit-subhead">Tam Soru</div>
        <p class="finding-text" style="font-size:0.88rem;">${esc(m.question)}</p>
      </div>
    `);
  }
  const sqs = rep?.sub_questions || [];
  if (sqs.length > 0) {
    const sqRows = sqs
      .map((sq) => {
        const cls =
          sq.status === "answered"
            ? "pill-answered"
            : sq.status === "conflicting"
              ? "pill-conflict"
              : sq.status === "partially_answered"
                ? "pill-demo"
                : "";
        return `<li class="sq-item"><span class="pill ${cls}">${esc(sq.status_label)}</span><span>${esc(sq.text)}</span></li>`;
      })
      .join("");
    auditChunks.push(`<div><div class="audit-subhead">Alt Sorular</div><ul class="sq-list">${sqRows}</ul></div>`);
  }
  const lims = rep?.limitations || [];
  if (lims.length > 0) {
    const limRows = lims
      .map(
        (l) =>
          `<li class="sq-item"><span class="pill">${esc(l.verdict_label)}</span><span><strong>${esc(l.statement)}</strong> — ${esc(l.rationale)}</span></li>`
      )
      .join("");
    auditChunks.push(`<div><div class="audit-subhead">Sınırlılıklar ve Elenen İddialar</div><ul class="lim-list">${limRows}</ul></div>`);
  }
  if (bib.length > 0) {
    const jumps = bib
      .map(
        (b) =>
          `<button type="button" class="jump-btn" data-jump-src="${esc(b.source_id)}">[${esc(b.source_id)}] ${esc(b.short_label)}</button>`
      )
      .join("");
    auditChunks.push(`<div><div class="audit-subhead">Kaynak Kartlarına Git</div><div class="source-jump-row">${jumps}</div></div>`);
  }
  els.auditBody.innerHTML = auditChunks.join("");
  els.auditDetails.open = false;

  els.auditBody.querySelectorAll("[data-jump-src]").forEach((btn) => {
    btn.addEventListener("click", () => {
      jumpToSourceCard(btn.dataset.jumpSrc);
    });
  });

  // Sekme 2: Kaynak Kartları
  const sources = bundle.sources || [];
  if (sources.length === 0) {
    els.sourceCards.innerHTML = `<div class="source-card"><p class="finding-note" style="margin:0;">Bu çalışmada kaydedilmiş kaynak bulunmuyor.</p></div>`;
  } else {
    els.sourceCards.innerHTML = sources
      .map((s) => {
        const openLink =
          s.has_external_url && s.url
            ? `<a class="source-link" href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">Kaynağı Aç</a>`
            : `<span class="pill pill-demo">Kurmaca</span>`;
        const excerptsHtml = (s.excerpts || [])
          .map((ch) => `<div class="quote-block">${esc(ch.excerpt_text)}</div>`)
          .join("");
        const dupNote = s.duplicate_of_source_id
          ? `<p class="finding-note">Bu kaynak ${esc(s.duplicate_of_source_id)} ile aynı içeriğe sahiptir.</p>`
          : "";
        return `
          <article class="source-card" id="source-card-${esc(s.source_id)}">
            <div class="source-card-top">
              <div>
                <h3 class="source-card-title">[${esc(s.source_id)}] ${esc(s.display_label)}</h3>
                <div class="source-card-meta">
                  ${esc(s.source_type_label)} · Yayın: ${esc(s.published_date_label)} · Uygunluk: ${s.relevance_score} · Güvenilirlik: ${s.credibility_score}
                </div>
              </div>
              <div>${openLink}</div>
            </div>
            ${dupNote}
            ${excerptsHtml}
          </article>
        `;
      })
      .join("");
  }

  const provs = bundle.provenance || [];
  els.sectionProvenance.classList.toggle("hidden", provs.length === 0);
  if (provs.length > 0) {
    els.claimSelect.innerHTML = provs
      .map((p) => `<option value="${esc(p.claim_id)}">${esc(p.claim_id)}: ${esc(p.statement)}</option>`)
      .join("");
    renderProvenanceDetail(provs[0].claim_id);
  }

  // Sekme 3: Süreç
  const durStr = m.duration_ms >= 1000 ? `${(m.duration_ms / 1000).toFixed(1)} sn` : `${m.duration_ms} ms`;
  els.processStats.innerHTML = `
    <div class="stat-card"><div class="stat-label">Durum</div><div class="stat-value">${esc(m.status_label)}</div></div>
    <div class="stat-card"><div class="stat-label">Kaynak</div><div class="stat-value">${m.source_count}</div></div>
    <div class="stat-card"><div class="stat-label">Arama</div><div class="stat-value">${m.search_calls}</div></div>
    <div class="stat-card"><div class="stat-label">Süre</div><div class="stat-value">${esc(durStr)}</div></div>
  `;

  const followups = (bundle.search_rounds || []).filter((sr) => sr.round_no > 1);
  els.processFollowups.innerHTML = followups
    .map(
      (fu) =>
        `<div class="finding-card" style="margin-bottom:16px;"><div class="finding-statement" style="margin-bottom:4px;">2. Tur Takip Araması</div><div class="finding-note" style="margin:0;">Sorgu: <strong>${esc(fu.query)}</strong>${fu.trigger_reason ? ` · ${esc(fu.trigger_reason)}` : ""}</div></div>`
    )
    .join("");

  els.processTimeline.innerHTML = (bundle.events || [])
    .map(
      (ev) => `
      <li class="timeline-item">
        <div class="timeline-head">
          <span>${esc(ev.node_label)}</span>
          <span class="timeline-sub">Tur ${ev.round_no} · ${ev.duration_ms} ms</span>
        </div>
        <p class="timeline-msg">${esc(ev.message)}</p>
      </li>
    `
    )
    .join("");

  setActiveTab("report");
  setView("report");
  renderHistory();
}

function jumpToSourceCard(sourceId) {
  setActiveTab("sources");
  const target = document.getElementById(`source-card-${sourceId}`);
  if (!target) return;
  document.querySelectorAll(".source-card.is-highlighted").forEach((c) => c.classList.remove("is-highlighted"));
  target.classList.add("is-highlighted");
  target.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function fetchRuns() {
  const res = await fetch("/api/runs");
  if (!res.ok) return;
  const data = await res.json();
  state.runs = data.runs || [];
  renderHistory();
}

async function loadRun(runId) {
  const res = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
  if (!res.ok) return;
  const bundle = await res.json();
  renderBundle(bundle);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function deleteRun(runId) {
  const res = await fetch(`/api/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
  if (!res.ok) return;
  state.runs = state.runs.filter((r) => r.run_id !== runId);
  if (state.activeBundle?.metadata?.run_id === runId) {
    state.activeBundle = null;
    setView("composer");
  }
  renderHistory();
}

async function clearAllRuns() {
  const res = await fetch("/api/runs", { method: "DELETE" });
  if (!res.ok) return;
  state.runs = [];
  state.activeBundle = null;
  setView("composer");
  renderHistory();
}

function appendOrUpdateLiveStep(ev) {
  const key = `${ev.node_name}_${ev.round_no}`;
  let li = els.liveSteps.querySelector(`[data-step-key="${key}"]`);
  if (!li) {
    li = document.createElement("li");
    li.className = "live-step";
    li.dataset.stepKey = key;
    li.innerHTML = `
      <span class="live-step-dot"></span>
      <div>
        <div class="live-step-title"></div>
        <div class="live-step-msg"></div>
      </div>
    `;
    els.liveSteps.appendChild(li);
  }
  li.classList.toggle("is-done", ev.status === "finish" || ev.status === "decision");
  li.classList.toggle("is-error", ev.status === "error");
  li.querySelector(".live-step-title").textContent = `${ev.node_label} · Tur ${ev.round_no}`;
  li.querySelector(".live-step-msg").textContent = ev.message;
}

async function handleResearchSubmit(ev) {
  ev.preventDefault();
  if (state.isRunning) return;

  const question = els.questionInput.value.trim();
  els.formError.classList.add("hidden");

  if (question.length < 10) {
    els.formError.textContent = "Lütfen en az 10 karakter uzunluğunda bir araştırma sorusu yazın.";
    els.formError.classList.remove("hidden");
    return;
  }

  state.isRunning = true;
  els.submitBtn.disabled = true;
  els.runningQuestion.textContent = question;
  els.liveSteps.innerHTML = "";
  setView("running");

  try {
    const response = await fetch("/api/research/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        mode: state.form.mode,
        depth: state.form.depth,
        language: state.form.language,
      }),
    });

    if (!response.ok) {
      let msg = "Araştırma başlatılamadı.";
      try {
        const errBody = await response.json();
        if (errBody?.detail) msg = errBody.detail;
      } catch {
        // yoksay
      }
      setView("composer");
      els.formError.textContent = msg;
      els.formError.classList.remove("hidden");
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let finalBundle = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const chunk of parts) {
        const line = chunk
          .split("\n")
          .find((l) => l.startsWith("data:"));
        if (!line) continue;
        const payload = JSON.parse(line.slice(5).trim());
        if (payload.type === "event") {
          appendOrUpdateLiveStep(payload);
        } else if (payload.type === "complete") {
          finalBundle = payload.data;
        } else if (payload.type === "error") {
          throw new Error(payload.message || "Araştırma sırasında hata oluştu.");
        }
      }
    }

    if (finalBundle) {
      await fetchRuns();
      renderBundle(finalBundle);
    } else {
      setView("composer");
    }
  } catch (err) {
    setView("composer");
    els.formError.textContent = err.message || "Bağlantı hatası oluştu.";
    els.formError.classList.remove("hidden");
  } finally {
    state.isRunning = false;
    els.submitBtn.disabled = false;
  }
}

function showComposer() {
  setView("composer");
  renderHistory();
  closeMobileSidebar();
  els.questionInput.focus();
}

function bindEvents() {
  els.brandHomeBtn.addEventListener("click", showComposer);
  els.newResearchBtn.addEventListener("click", showComposer);
  els.mobileNewBtn.addEventListener("click", showComposer);

  els.openSidebarBtn.addEventListener("click", () => {
    els.sidebar.classList.add("is-open");
    els.backdrop.classList.remove("hidden");
  });
  els.closeSidebarBtn.addEventListener("click", closeMobileSidebar);
  els.backdrop.addEventListener("click", closeMobileSidebar);

  els.historySearch.addEventListener("input", renderHistory);
  els.clearHistoryBtn.addEventListener("click", clearAllRuns);
  els.researchForm.addEventListener("submit", handleResearchSubmit);

  els.questionInput.addEventListener("keydown", (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") {
      ev.preventDefault();
      els.researchForm.requestSubmit();
    }
  });

  document.querySelectorAll(".seg-group").forEach((group) => {
    const key = group.dataset.control;
    group.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.form[key] = btn.dataset.value;
        syncSegmentedControls();
      });
    });
  });

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
  });

  els.claimSelect.addEventListener("change", () => {
    renderProvenanceDetail(els.claimSelect.value);
  });
}

async function init() {
  bindEvents();
  const cfgRes = await fetch("/api/config");
  if (cfgRes.ok) {
    state.config = await cfgRes.json();
    state.form.mode = state.config.default_mode || "demo";
    if (state.config.has_tavily_key) {
      els.envStatusDot.classList.add("is-live");
      els.envStatusText.textContent = "Canlı Web hazır";
    } else {
      els.envStatusDot.classList.add("is-demo");
      els.envStatusText.textContent = "Çevrimdışı Demo";
    }
    renderPresets();
    syncSegmentedControls();
  }
  await fetchRuns();
  setView("composer");
}

init();
