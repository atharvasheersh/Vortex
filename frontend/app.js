const paths = {
  search: '<circle cx="10.8" cy="10.8" r="6.7"/><path d="m16 16 4.5 4.5"/>',
  layers:
    '<path d="m12 3 9 5-9 5-9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/>',
  chart: '<path d="M4 4v16h17M8 15l4-5 4 2 5-7"/>',
  workflow:
    '<rect x="8" y="2" width="8" height="5" rx="1"/><rect x="2" y="17" width="7" height="5" rx="1"/><rect x="15" y="17" width="7" height="5" rx="1"/><path d="M12 7v5M5.5 17v-5h13v5"/>',
  orbit:
    '<ellipse cx="12" cy="12" rx="10" ry="5" transform="rotate(-35 12 12)"/><ellipse cx="12" cy="12" rx="10" ry="5" transform="rotate(45 12 12)"/><circle cx="12" cy="12" r="1"/>',
  "arrow-up-right": '<path d="M6 18 18 6M6 6h12v12"/>',
  "arrow-right": '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  shield:
    '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6Z"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
  box: '<path d="m12 3 9 5v9l-9 5-9-5V8Zm0 10v9M3 8l9 5 9-5M7.5 5.5l9 5"/>',
  github:
    '<path d="M9 19c-4 1-4-2-5-2m14 5v-4a3.5 3.5 0 0 0-1-2.7c3.3-.4 6.7-1.6 6.7-7a5.4 5.4 0 0 0-1.5-3.7 5 5 0 0 0-.1-3.6s-1.3-.4-4.1 1.5a14 14 0 0 0-7 0C8.2.6 6.9 1 6.9 1a5 5 0 0 0-.1 3.6 5.4 5.4 0 0 0-1.5 3.7c0 5.4 3.4 6.6 6.7 7A3.5 3.5 0 0 0 11 18v4" transform="translate(-2 0) scale(.96)"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
  crosshair:
    '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3"/>',
  terminal: '<path d="m5 6 5 6-5 6m8 0h6"/>',
  database:
    '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
  zap: '<path d="m13 2-9 12h7l-1 8 10-13h-8Z"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.2"/>',
  download: '<path d="M12 3v12m-4-4 4 4 4-4M4 16v5h16v-5"/>',
  lightbulb: '<path d="M9 18h6m-6 3h6M8.5 15a7 7 0 1 1 7 0L15 18H9Z"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
  alert: '<path d="m12 3 10 18H2Z"/><path d="M12 9v5m0 3v.2"/>',
  file: '<path d="M14 2H5v20h14V7Zm0 0v6h5M8 12h8m-8 4h6"/>',
  x: '<path d="m6 6 12 12M6 18 18 6"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>',
};
const icon = (name) =>
  `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.file}</svg>`;
const $ = (selector) => document.querySelector(selector);
const esc = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
const number = (value) => new Intl.NumberFormat("en-US").format(value);
const duration = (ms) =>
  ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(3)} ms`;
const state = {
  bootstrap: null,
  result: null,
  benchmarks: null,
  selectedSize: 10000000,
  inspectedProbe: null,
  view: "playground",
  request: 0,
  controller: null,
  busy: false,
};
const titles = {
  playground: "Search playground",
  collection: "Collection",
  benchmarks: "Benchmarks",
  architecture: "Architecture",
};
let history = [];
try {
  history = JSON.parse(localStorage.getItem("vortex.query-history") || "[]")
    .filter((q) => typeof q === "string")
    .slice(0, 6);
} catch {
  /* Storage can be unavailable in private browsing. */
}

function fillIcons(container = document) {
  container.querySelectorAll("[data-icon]").forEach((el) => {
    el.innerHTML = icon(el.dataset.icon);
  });
}
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => {
    $("#toast").hidden = true;
  }, 3000);
}
async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const payload = await response.json();
  if (!response.ok)
    throw new Error(payload.error || "The request could not be completed.");
  return payload;
}
function renderHistory() {
  $("#recent-queries").innerHTML = history.length
    ? history
        .map(
          (query) =>
            `<button class="recent-query" data-history="${esc(query)}">${icon("clock")}<span>${esc(query)}</span></button>`,
        )
        .join("")
    : "<p>Your searches will appear here.</p>";
  $("#clear-history").disabled = !history.length;
}
function remember(query) {
  history = [query, ...history.filter((entry) => entry !== query)].slice(0, 6);
  try {
    localStorage.setItem("vortex.query-history", JSON.stringify(history));
  } catch {
    /* Search remains available without persistence. */
  }
  renderHistory();
}
function settings() {
  return {
    query: $("#query").value.trim(),
    mode: $("#query-form input[name=mode]:checked").value,
    k: Number($("#top-k").value),
    nprobe: Number($("#nprobe").value),
  };
}
function updateControls() {
  const data = settings();
  const maximum = state.bootstrap?.collection.nlist || 8;
  $("#nprobe").disabled = data.mode === "exact";
  $("#probe-value").textContent = `${data.nprobe} / ${maximum}`;
  $("#probe-note").textContent =
    data.mode === "exact"
      ? "Exact search visits every vector."
      : "More lists usually improve neighbor recall and increase the number of candidates.";
  renderResultSummary();
}
function renderResultSummary() {
  if (!state.result) return;
  const result = state.result;
  const current = settings();
  const stale =
    current.query !== result.query ||
    current.mode !== result.mode ||
    current.k !== result.k ||
    (current.mode === "ivf" && current.nprobe !== result.nprobe);
  $("#result-summary").innerHTML =
    `<strong>${result.hits.length}</strong> neighbors <span class="summary-separator">·</span><strong>${duration(result.native_ms)}</strong> native search <span class="summary-separator">·</span>${result.mode === "exact" ? "Exact" : `IVF · ${result.nprobe} probe${result.nprobe === 1 ? "" : "s"}`}${stale ? '<span class="badge neutral">SETTINGS CHANGED · RUN TO APPLY</span>' : ""}`;
}
function highlight(text, query) {
  const terms = [
    ...new Set(
      (query.toLowerCase().match(/[a-z0-9]{3,}/g) || []).filter(
        (word) =>
          !["the", "how", "does", "what", "are", "and", "for", "with"].includes(
            word,
          ),
      ),
    ),
  ];
  return terms.length
    ? esc(text).replace(
        new RegExp(`\\b(${terms.join("|")})\\b`, "gi"),
        "<mark>$1</mark>",
      )
    : esc(text);
}
function renderResults() {
  const result = state.result;
  $("#results-count").textContent = result.hits.length;
  $("#export-results").disabled = false;
  renderResultSummary();
  $("#results").innerHTML =
    result.hits
      .map(
        (hit) =>
          `<article class="result-card"><span class="rank">${String(hit.rank).padStart(2, "0")}</span><div class="result-body"><button class="result-title" data-document="${hit.id}">${esc(hit.title)}</button><div class="result-meta">${icon("file")}<span>${esc(hit.source)}:${hit.line}</span><span>·</span><span>${hit.words} words</span></div><p class="result-text">${highlight(hit.text, result.query)}</p></div><div class="score-column"><strong>${hit.score.toFixed(4)}</strong><small>cosine</small><div class="score-bar"><svg viewBox="0 0 100 3" aria-hidden="true"><rect width="100" height="3" rx="1.5" fill="#edf3e6"/><rect width="${Math.max(0, Math.min(100, hit.score * 100))}" height="3" rx="1.5" fill="#afce8a"/></svg></div></div></article>`,
      )
      .join("") ||
    '<div class="empty-state"><h3>No candidates in the selected lists</h3><p>Increase the number of probes or switch to exact search.</p></div>';
  $("#query-inspector").innerHTML =
    `<div class="trace-primary"><span>Native search</span><strong>${result.native_ms.toFixed(3)}<small>ms</small></strong></div>${[
      [
        "layers",
        "Vectors scanned",
        `${number(result.scanned)} / ${number(result.total_vectors)}`,
      ],
      [
        "crosshair",
        `Recall@${result.k} vs exact`,
        `${(result.recall_at_k * 100).toFixed(1)}%`,
      ],
      [
        "zap",
        "Search strategy",
        result.mode === "exact"
          ? "Exhaustive"
          : `IVF · ${result.nprobe} list${result.nprobe === 1 ? "" : "s"}`,
      ],
      ["grid", "Encode query", duration(result.embedding_ms)],
      ["file", "Resolved evidence", `${result.hits.length} documents`],
    ]
      .map(
        ([glyph, label, value]) =>
          `<div class="trace-row"><span>${icon(glyph)}${label}</span><strong>${value}</strong></div>`,
      )
      .join("")}`;
  renderMap();
}
function renderMap() {
  if (!state.bootstrap) return;
  const matched = new Set(state.result?.hits.map((hit) => hit.id) || []);
  const query = state.result?.query_point;
  const extent =
    Math.max(1, query ? Math.abs(query.x) : 0, query ? Math.abs(query.y) : 0) *
    1.1;
  const x = (value) => 28 + ((value / extent + 1) / 2) * 244;
  const y = (value) => 206 - ((value / extent + 1) / 2) * 178;
  $("#vector-map").innerHTML =
    `<svg viewBox="0 0 300 234" role="group" aria-label="Document vector projection. Use left and right arrow keys to explore points, and Enter to inspect a document."><path d="M28 117h244M150 25v185" stroke="#e6ecdf" stroke-dasharray="3 4"/>${state.bootstrap.projection.map((point, pointIndex) => `<circle cx="${x(point.x)}" cy="${y(point.y)}" r="${matched.has(point.id) ? 4.8 : 3.1}" fill="${matched.has(point.id) ? "#91b85e" : "#cbd8bb"}" stroke="${matched.has(point.id) ? "#fff" : "none"}" stroke-width="1.5" opacity="${matched.has(point.id) ? 1 : 0.55}" data-document="${point.id}" tabindex="${pointIndex === 0 ? 0 : -1}" role="button" aria-label="Inspect ${esc(state.bootstrap.documents.find((d) => d.id === point.id).title)}"><title>${esc(state.bootstrap.documents.find((d) => d.id === point.id).title)}</title></circle>`).join("")}${query ? `<path d="m${x(query.x)} ${y(query.y) - 5} 5 5-5 5-5-5Z" fill="#304a2c" stroke="#fff" stroke-width="1.5"/><text x="${Math.min(265, x(query.x) + 9)}" y="${y(query.y) + 3}" fill="#667e51" font-size="8" font-family="monospace">query</text>` : ""}<text x="250" y="224" class="map-axis-label">PC 1 →</text><text x="8" y="18" class="map-axis-label">PC 2 ↑</text></svg>`;
}
async function search(reveal = false) {
  if (!state.bootstrap) return;
  const payload = settings();
  if (!payload.query) {
    $("#query").reportValidity();
    return;
  }
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  const request = ++state.request;
  state.busy = true;
  $("#search-button").disabled = true;
  $("#search-button").innerHTML = `Searching…${icon("search")}`;
  $(".results-panel").setAttribute("aria-busy", "true");
  try {
    const result = await api("/api/search", {
      method: "POST",
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (request !== state.request) return;
    state.result = result;
    renderResults();
    remember(result.query);
    if (reveal && window.innerWidth < 1000) {
      $(".results-panel").scrollIntoView({
        block: "start",
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "instant"
          : "smooth",
      });
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      $("#result-summary").textContent =
        `Search failed: ${error.message}${state.result ? " · Previous results remain below." : ""}`;
      toast(error.message);
    }
  } finally {
    if (request === state.request) {
      state.busy = false;
      $("#search-button").disabled = false;
      $("#search-button").innerHTML = `Run search${icon("arrow-right")}`;
      $(".results-panel").setAttribute("aria-busy", "false");
    }
  }
}
function openDocument(id) {
  const doc = state.bootstrap?.documents.find((item) => item.id === id);
  if (!doc) return;
  $("#dialog-content").innerHTML =
    `<span class="badge neutral">FP32 · 256D · DOCUMENT CHUNK</span><h2 id="dialog-title">${esc(doc.title)}</h2><p class="document-source">${esc(doc.source)} · line ${doc.line} · ${doc.words} words</p><pre>${esc(doc.text)}</pre><a class="button secondary" href="/source/${encodeURI(doc.source)}" target="_blank" rel="noreferrer">Open source document ${icon("arrow-up-right")}</a><p class="doc-id">Stable vector ID<br>${doc.id}</p>`;
  $("#document-dialog").showModal();
}
function renderDocuments() {
  if (!state.bootstrap) return;
  const term = $("#document-filter").value.toLowerCase().trim();
  const docs = state.bootstrap.documents.filter((doc) =>
    `${doc.title} ${doc.source} ${doc.text}`.toLowerCase().includes(term),
  );
  $("#document-count").textContent = docs.length;
  $("#documents-empty").hidden = docs.length > 0;
  $("#documents-body").innerHTML = docs
    .map(
      (doc) =>
        `<tr data-document="${doc.id}"><td><button class="result-title" data-document="${doc.id}">${esc(doc.title)}</button></td><td>${esc(doc.source)}:${doc.line}</td><td>${doc.words}</td><td title="${doc.id}">${doc.id.slice(0, 9)}…</td><td>${icon("chevron")}</td></tr>`,
    )
    .join("");
  const c = state.bootstrap.collection;
  $("#collection-overview").innerHTML =
    `<span>${icon("database")}</span><div><h2>${esc(c.name)}</h2><p>${number(c.vectors)} chunks from ${c.sources} repository documents · stable 64-bit IDs</p></div><div class="collection-tags"><span class="badge neutral">256D FP32</span><span class="badge neutral">COSINE</span><span class="badge neutral">IN-MEMORY</span><span class="badge">TOKEN-HASH FIXTURE</span></div>`;
}
async function loadBenchmarks() {
  if (state.benchmarks) {
    renderBenchmarks();
    return;
  }
  try {
    state.benchmarks = await api("/api/benchmarks");
    renderBenchmarks();
  } catch (error) {
    $("#benchmark-content").innerHTML =
      `<div class="panel empty-state"><h3>Evidence unavailable</h3><p>${esc(error.message)}</p><button class="button secondary" id="retry-benchmarks">Try again</button></div>`;
  }
}
function renderBenchmarks() {
  const data = state.benchmarks;
  if (!data) return;
  const run =
    data.runs.find((row) => row.vectors === state.selectedSize) || data.runs[0];
  state.selectedSize = run.vectors;
  const operating = run.probes.find((p) => p.nprobe === run.nprobe);
  const inspected =
    run.probes.find((p) => p.nprobe === state.inspectedProbe) || operating;
  state.inspectedProbe = inspected.nprobe;
  const diagnostics = data.diagnostics.find(
    (entry) => entry.size === run.vectors,
  );
  const gaps = diagnostics.timing_gaps;
  const interrupted = (probe) => gaps.some((gap) => gap.nprobe === probe);
  const maximumMs =
    Math.max(
      ...run.probes.map((p) => p.median_round_p95_ms),
      run.exact.median_round_p95_ms,
    ) * 1.15;
  const plotWidth = window.innerWidth <= 720 ? 248 : 492;
  const chartWidth = window.innerWidth <= 720 ? 330 : 580;
  const cx = (ms) => 48 + (ms / maximumMs) * plotWidth;
  const cy = (recall) => 240 - recall * 190;
  const ticks = Array.from({ length: 5 }, (_, i) => i / 4);
  const chart = `<svg viewBox="0 0 ${chartWidth} 290" role="group" aria-label="Recorded IVF latency and neighbor recall. Choose a probe to inspect its measurements.">${ticks.map((r) => `<path d="M48 ${cy(r)}h${plotWidth}" stroke="#edf1e5"/><text x="36" y="${cy(r) + 3}" text-anchor="end" class="chart-label">${Math.round(r * 100)}%</text>`).join("")}${ticks.map((t) => `<text x="${cx(t * maximumMs)}" y="260" text-anchor="middle" class="chart-label">${((t * maximumMs) / 1000).toFixed(1)}</text>`).join("")}<text x="48" y="25" class="chart-title">RECALL@10</text><text x="${48 + plotWidth / 2}" y="284" text-anchor="middle" class="chart-title">MEDIAN ROUND P95 LATENCY (SECONDS)</text><polyline points="${run.probes.map((p) => `${cx(p.median_round_p95_ms)},${cy(p.recall_at_k)}`).join(" ")}" fill="none" stroke="#a9b6ca" stroke-width="1.8"/>${run.probes.map((p) => `<g data-probe="${p.nprobe}" tabindex="0" role="button" aria-label="Inspect ${p.nprobe} probes${interrupted(p.nprobe) ? ", timing interrupted" : ""}"><circle cx="${cx(p.median_round_p95_ms)}" cy="${cy(p.recall_at_k)}" r="14" fill="transparent"/><circle cx="${cx(p.median_round_p95_ms)}" cy="${cy(p.recall_at_k)}" r="${p.nprobe === inspected.nprobe ? 8 : 5}" fill="${interrupted(p.nprobe) ? "#c3a36b" : p.nprobe === inspected.nprobe ? "#91b65b" : "#a2b0c1"}" stroke="#fff" stroke-width="2"/><text x="${cx(p.median_round_p95_ms) + 8}" y="${cy(p.recall_at_k) - 9}" class="chart-label">${p.nprobe}${interrupted(p.nprobe) ? "†" : ""}</text></g>`).join("")}<circle cx="${cx(run.exact.median_round_p95_ms)}" cy="${cy(1)}" r="4" fill="#384a32"/><text x="${cx(run.exact.median_round_p95_ms) + 8}" y="${cy(1) + 15}" class="chart-label">Exact</text></svg>`;
  $("#benchmark-content").innerHTML =
    `<div class="benchmark-control"><label for="benchmark-size">Collection size</label><select id="benchmark-size">${data.runs.map((r) => `<option value="${r.vectors}" ${r.vectors === run.vectors ? "selected" : ""}>${number(r.vectors)} vectors</option>`).join("")}</select><span class="badge neutral">ARCHIVED RUN</span></div><div class="metrics-grid"><article class="metric-card"><div class="metric-top">Snapshot size ${icon("database")}</div><strong>${(run.snapshot_bytes / 1e9).toFixed(2)}<span class="metric-unit">GB</span></strong><span class="metric-caption">One mapped collection</span></article><article class="metric-card"><div class="metric-top">Exact search p95 ${icon("crosshair")}</div><strong>${(run.exact_median_round_p95_ms / 1000).toFixed(2)}<span class="metric-unit">s</span></strong><span class="metric-caption">All ${number(run.vectors)} vectors scanned</span></article><article class="metric-card"><div class="metric-top">Selected IVF recall ${icon("zap")}</div><strong>${(run.ivf_recall_at_k * 100).toFixed(1)}<span class="metric-unit">%</span></strong><span class="metric-caption">${run.nprobe} probes · Recall@10</span></article><article class="metric-card"><div class="metric-top">Construction time ${icon("clock")}</div><strong>${run.construction_seconds.toFixed(1)}<span class="metric-unit">s</span></strong><span class="metric-caption">Streamed snapshot construction</span></article></div><div class="benchmark-layout"><section class="panel"><div class="panel-heading"><h2>Recall meets latency</h2><span class="badge neutral">SELECT A PROBE</span></div><div class="benchmark-chart">${chart}<p class="benchmark-footnote">${run.query_count} distinct queries × ${run.repeats} shuffled rounds. Exact and IVF share the same data. The operating point is the first tested probe with recall ≥95% on these measured queries.</p></div></section><section class="panel"><div class="panel-heading"><h2>Probe inspector</h2><span class="badge ${interrupted(inspected.nprobe) ? "neutral" : ""}">${inspected.nprobe} LISTS</span></div><div class="benchmark-selected"><h3>${inspected.nprobe === run.nprobe ? "Selected operating point" : "Recorded probe configuration"}</h3><strong>${(inspected.recall_at_k * 100).toFixed(1)}%</strong><p>nearest-neighbor Recall@10</p><div class="trace-row"><span>p95 latency</span><strong>${duration(inspected.median_round_p95_ms)}${interrupted(inspected.nprobe) ? " †" : ""}</strong></div><div class="trace-row"><span>Mean candidates</span><strong>${number(Math.round(inspected.mean_scanned))}</strong></div><div class="trace-row"><span>Requested resident limit</span><strong>1536 MiB</strong></div><div class="trace-row"><span>Timing diagnostic</span><strong>${interrupted(inspected.nprobe) ? "Gap recorded †" : "No flagged gap"}</strong></div></div></section></div><section class="panel"><div class="panel-heading"><h2>Full probe sweep</h2><span class="subtle-label">SQUARED L2 · 64D · FP32</span></div><div class="table-scroll"><table class="benchmark-table"><thead><tr><th>PROBES</th><th>RECALL@10</th><th>P95 LATENCY</th><th>MEAN CANDIDATES</th><th>STATUS</th></tr></thead><tbody>${run.probes.map((p) => `<tr class="${p.nprobe === inspected.nprobe ? "selected-row" : ""}"><td><button class="text-button" data-probe="${p.nprobe}">${p.nprobe} list${p.nprobe === 1 ? "" : "s"} ${icon("arrow-right")}</button></td><td>${(p.recall_at_k * 100).toFixed(1)}%</td><td>${duration(p.median_round_p95_ms)}${interrupted(p.nprobe) ? " †" : ""}</td><td>${number(Math.round(p.mean_scanned))}</td><td>${interrupted(p.nprobe) ? "Timing gap †" : p.nprobe === run.nprobe ? "Selected ≥95%" : "Recorded"}</td></tr>`).join("")}</tbody></table></div></section>${gaps.length ? `<div class="warning-note"><span>${icon("alert")}</span><div><strong>One all-list latency sample was interrupted.</strong><p>The 32-probe run includes ${(gaps[0].wall_ms / 60000).toFixed(1)} minutes wall time versus ${gaps[0].process_cpu_seconds.toFixed(1)} seconds CPU. The cause is unconfirmed. Every sample is retained; † marks a condition that is not a clean latency result. Exact and the selected ${run.nprobe}-probe condition have no flagged gaps.</p></div></div>` : ""}<div class="benchmark-sources"><span>Requested 1536 MiB resident limit · OS page cache not flushed · synthetic capacity, not RAG answer quality</span><a href="/source/docs/capacity_10m.md" target="_blank" rel="noreferrer">Read the complete methodology ${icon("arrow-up-right")}</a></div>`;
}
function setMenu(open) {
  const wasOpen = $("#sidebar").classList.contains("open");
  const mobile = window.innerWidth <= 720;
  const modal = mobile && open;
  $("#sidebar").classList.toggle("open", open);
  $("#sidebar").inert = mobile && !open;
  $(".app-shell").inert = modal;
  $(".skip-link").inert = modal;
  if (modal) {
    $("#sidebar").setAttribute("role", "dialog");
    $("#sidebar").setAttribute("aria-modal", "true");
    $("#sidebar").setAttribute("aria-label", "Workspace navigation");
    $("#close-menu").focus({ preventScroll: true });
  } else {
    $("#sidebar").removeAttribute("role");
    $("#sidebar").removeAttribute("aria-modal");
    $("#sidebar").removeAttribute("aria-label");
  }
  $("#menu-toggle").setAttribute("aria-expanded", String(open));
  $("#nav-backdrop").hidden = !open;
  if (!open && wasOpen && window.innerWidth <= 720) $("#menu-toggle").focus();
}
function navigate() {
  const previous = state.view;
  const next = location.hash.replace("#", "") || "playground";
  state.view = Object.hasOwn(titles, next) ? next : "playground";
  document.querySelectorAll(".view").forEach((view) => {
    view.hidden = view.id !== `view-${state.view}`;
  });
  document.querySelectorAll("[data-nav]").forEach((link) => {
    const current = link.dataset.nav === state.view;
    link.classList.toggle("active", current);
    if (current) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  $("#breadcrumb").textContent = titles[state.view];
  document.title = `Vortex · ${titles[state.view]}`;
  if (previous !== state.view)
    window.scrollTo({ top: 0, left: 0, behavior: "instant" });
  setMenu(false);
  if (state.view === "benchmarks") loadBenchmarks();
  if (state.view === "collection") renderDocuments();
}
async function connect() {
  $("#connection").className = "connection";
  $("#connection").innerHTML = "<i></i>Connecting";
  try {
    state.bootstrap = await api("/api/bootstrap");
    const { collection, engine, suggestions } = state.bootstrap;
    $("#global-error").hidden = true;
    $("#connection").className = "connection connected";
    $("#connection").innerHTML = "<i></i>Engine connected";
    $("#metric-vectors").textContent = number(collection.vectors);
    $("#metric-partitions").textContent = collection.nlist;
    $("#nav-count").textContent = collection.vectors;
    $("#engine-version").textContent =
      `VORTEX ${engine.version} · ${engine.avx2 ? "AVX2" : "SCALAR"} · NATIVE`;
    $("#nprobe").max = collection.nlist;
    $("#nprobe").value = Math.min(4, collection.nlist);
    $("#suggestions").innerHTML = suggestions
      .slice(1)
      .map(
        (query) =>
          `<button type="button" class="suggestion" data-suggestion="${esc(query)}">${esc(query)}</button>`,
      )
      .join("");
    if (!$("#query").value) $("#query").value = suggestions[0];
    $("#search-button").disabled = false;
    renderDocuments();
    renderMap();
    updateControls();
    await search();
  } catch (error) {
    state.bootstrap = null;
    $("#connection").className = "connection offline";
    $("#connection").innerHTML = "<i></i>Engine offline";
    $("#global-error-text").textContent = error.message;
    $("#global-error").hidden = false;
    $("#search-button").disabled = true;
  }
}

$("#query-form").addEventListener("submit", (event) => {
  event.preventDefault();
  search(true);
});
$("#query-form").addEventListener("input", updateControls);
$("#query-form").addEventListener("change", updateControls);
$("#document-filter").addEventListener("input", renderDocuments);
$("#retry-connection").addEventListener("click", connect);
$("#close-document").addEventListener("click", () =>
  $("#document-dialog").close(),
);
$("#menu-toggle").addEventListener("click", () => {
  setMenu(!$("#sidebar").classList.contains("open"));
});
$("#close-menu").addEventListener("click", () => {
  setMenu(false);
  $("#menu-toggle").focus();
});
$("#nav-backdrop").addEventListener("click", () => setMenu(false));
$(".skip-link").addEventListener("click", (event) => {
  event.preventDefault();
  $("#main").focus();
});
$("#clear-history").addEventListener("click", () => {
  history = [];
  try {
    localStorage.removeItem("vortex.query-history");
  } catch {
    /* Optional local storage. */
  }
  renderHistory();
  toast("Recent query history cleared");
});
$("#export-results").addEventListener("click", () => {
  if (!state.result) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(state.result, null, 2)], {
      type: "application/json",
    }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = "vortex-search-results.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("Search results exported as JSON");
});
document.addEventListener("click", (event) => {
  const doc = event.target.closest("[data-document]");
  if (doc) {
    openDocument(doc.dataset.document);
    return;
  }
  const suggested = event.target.closest("[data-suggestion],[data-history]");
  if (suggested) {
    if (state.view !== "playground") {
      location.hash = "playground";
      navigate();
    }
    $("#query").value =
      suggested.dataset.suggestion || suggested.dataset.history;
    setMenu(false);
    $("#query").focus();
    search(true);
    return;
  }
  const probe = event.target.closest("[data-probe]");
  if (probe) {
    state.inspectedProbe = Number(probe.dataset.probe);
    const fromChart = probe.tagName.toLowerCase() === "g";
    renderBenchmarks();
    $(
      `#benchmark-content ${fromChart ? "svg" : ".benchmark-table"} [data-probe="${state.inspectedProbe}"]`,
    )?.focus({ preventScroll: true });
    if (!fromChart)
      $(".benchmark-selected").scrollIntoView({
        block: "nearest",
        behavior: "smooth",
      });
    return;
  }
  if (event.target.closest("#retry-benchmarks")) loadBenchmarks();
});
document.addEventListener("change", (event) => {
  if (event.target.id === "benchmark-size") {
    state.selectedSize = Number(event.target.value);
    state.inspectedProbe = null;
    renderBenchmarks();
    $("#benchmark-size").focus({ preventScroll: true });
  }
});
document.addEventListener("keydown", (event) => {
  if (
    event.key === "Tab" &&
    $("#sidebar").getAttribute("aria-modal") === "true"
  ) {
    const links = [
      ...$("#sidebar").querySelectorAll("a[href],button:not(:disabled)"),
    ];
    const first = links[0],
      last = links[links.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    }
    if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    if (state.view !== "playground") {
      location.hash = "playground";
      navigate();
    }
    setMenu(false);
    $("#query").focus();
    $("#query").select();
  }
  if (
    (event.ctrlKey || event.metaKey) &&
    event.key === "Enter" &&
    state.view === "playground"
  ) {
    event.preventDefault();
    search(true);
  }
  const node = event.target.closest?.("svg [data-document],svg [data-probe]");
  if (
    node?.matches("#vector-map circle") &&
    ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)
  ) {
    event.preventDefault();
    const points = [
      ...document.querySelectorAll("#vector-map circle[data-document]"),
    ];
    const current = points.indexOf(node);
    const next =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? points.length - 1
          : (current + (event.key === "ArrowRight" ? 1 : -1) + points.length) %
            points.length;
    points[next].focus();
  }
  if (node && ["Enter", " "].includes(event.key)) {
    event.preventDefault();
    node.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  }
  if (event.key === "Escape") {
    setMenu(false);
  }
});
document.addEventListener("focusin", (event) => {
  if (event.target.matches?.("#vector-map circle[data-document]")) {
    document
      .querySelectorAll("#vector-map circle[data-document]")
      .forEach((point) =>
        point.setAttribute("tabindex", point === event.target ? "0" : "-1"),
      );
  }
});
window.matchMedia("(max-width: 720px)").addEventListener("change", () => {
  setMenu(false);
  if (state.benchmarks) renderBenchmarks();
});
window.addEventListener("hashchange", navigate);
fillIcons();
renderHistory();
navigate();
connect();
