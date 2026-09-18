/* Wayfinder frontend — vanilla JS, no build step.
 * Consumes:  GET  /api/health, /api/kb, /api/memory
 *           POST /api/query   (SSE stream: agent events or search_results)
 *           POST /api/search  (JSON, direct hard search)
 */
"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  mode: "agent",          // "agent" | "direct"
  running: false,
  traceCount: 0,
  answerSources: [],      // for [n] citation mapping
  controller: null,       // AbortController for the live SSE run (CP 6.1 L6 — Stop)
};

/* ================================================================
 * helpers
 * ================================================================ */
function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/** HTML-attribute-safe escape. Unlike escapeHtml it ALSO escapes the single
    quote — required for values placed in single-quoted attributes like
    data-summary='…'. A stray ' in the JSON (e.g. a place named "Angel's")
    would otherwise terminate the attribute early, JSON.parse would throw, and
    the dependent render/calc would silently never run. */
function attrEscape(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

function tierBadge(tier) {
  const t = (tier || "secondary").toLowerCase();
  const labels = { primary: "core", secondary: "supporting", tertiary: "community", web: "live web" };
  return `<span class="tier tier-${t}">${labels[t] || t}</span>`;
}

/** inline markdown: **bold**, _italic_, [n] citations -> clickable */
function renderInline(text) {
  let s = escapeHtml(text);
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|\s)_([^_\n]+)_/g, "$1<em>$2</em>");
  s = s.replace(/\[(\d+)\]/g, (m, n) =>
    `<sup class="cite" data-cite="${n}" title="Jump to source ${n}">[${n}]</sup>`);
  return s;
}

/** mini-markdown for answer sections: ## headers, • bullets, _notes_ */
function renderSection(md) {
  const out = [];
  let list = null;
  const flush = () => { if (list !== null) { out.push(`<ul>${list}</ul>`); list = null; } };
  for (const raw of String(md).split("\n")) {
    const line = raw.trim();
    if (!line) { flush(); continue; }
    if (line.startsWith("## ")) { flush(); out.push(`<h3>${renderInline(line.slice(3))}</h3>`); }
    else if (line.startsWith("• ")) { if (list === null) list = ""; list += `<li>${renderInline(line.slice(2))}</li>`; }
    else if (/^_.+_$/ .test(line)) { flush(); out.push(`<p class="note">${renderInline(line)}</p>`); }
    else { flush(); out.push(`<p>${renderInline(line)}</p>`); }
  }
  flush();
  return out.join("");
}

/* Cross-source research leads — one dropdown per source (learn-more link + quote + note). */
const LEAD_NOTES = {
  reddit:      "📌 This is the Reddit post link — open it to read the thread. Discovery only: never fetched behind a login wall.",
  instagram:   "📌 This is the Instagram post link — open it to view. Discovery only: never fetched behind a login wall.",
  tripadvisor: "Lead only — not live inventory or pricing. Open the listing to verify details.",
  expedia:     "Lead only — not live inventory or pricing. Open the listing to check current rates and availability.",
  wikivoyage:  "Reference guide — open the article to verify details.",
  wikipedia:   "Reference article — open it to verify details.",
};

function renderLeads(leads) {
  const items = leads.map((lead) => {
    const type = String(lead.source_type || "web").toLowerCase();
    const label = escapeHtml(lead.source_label || type);
    const url = lead.url || "";
    let host = url;
    if (url) { try { host = new URL(url).host.replace(/^www\./, ""); } catch { /* keep raw url */ } }
    const linkRow = url
      ? `<a class="lead-link" href="${escapeHtml(url)}" target="_blank" rel="noopener">🔗 Learn more — open ${escapeHtml(host)} ↗</a>`
      : `<span class="lead-nolink muted">No public link for this lead.</span>`;
    const quote = lead.quote
      ? `<blockquote class="lead-quote">“${escapeHtml(lead.quote)}”</blockquote>`
      : "";
    const cite = lead.citation
      ? `<sup class="cite" data-cite="${lead.citation}" title="Jump to source ${lead.citation}">[${lead.citation}]</sup>`
      : "";
    const note = `<p class="lead-note">${escapeHtml(LEAD_NOTES[type] || "Open the source to verify details.")}</p>`;
    return `<details class="lead lead-${type}">
      <summary>
        <span class="lead-badge">${label}</span>
        <span class="lead-title">${escapeHtml(lead.title || "research lead")}</span>
        ${cite}
        <span class="lead-caret" aria-hidden="true">▾</span>
      </summary>
      <div class="lead-body">${linkRow}${quote}${note}</div>
    </details>`;
  }).join("");
  return `<section class="leads-block">
    <h3>🗂️ Cross-source research leads</h3>
    <p class="leads-hint">Open a source to learn more — each dropdown links the public page, with a quote when one was found.</p>
    ${items}
    <p class="note">Public discovery only: Reddit and Instagram results are links/snippets, never login-wall scraping. Tripadvisor and Expedia are leads, not live inventory or pricing. Verify each source before acting.</p>
  </section>`;
}

/* Chance of rain for a month: share of days with >= 0.1 mm of rain in the
   stored 2023-2025 climate normals (Open-Meteo wet-day definition). Pure
   derivation of real stored data — never a forecast, never invented (CP 1.1/3.1). */
const SEASON_MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
function seasonRainProb(m) {
  const d = SEASON_MONTH_DAYS[(m.m || 1) - 1];
  return Math.min(100, Math.round((100 * (m.rain_days || 0)) / d));
}

/* Weather chart — inline SVG bar chart of chance of rain per month (no external
   libraries, key-free, works offline). Bars are colored by the same ideal/
   shoulder/off bands as the chips, the best month is starred, and each bar has
   a hover tooltip with the underlying numbers. */
function renderRainChart(s) {
  const months = (s.months || []).slice().sort((a, b) => a.m - b.m);
  if (!months.length) return "";
  const W = 680, H = 200, PADL = 44, PADR = 10, PADT = 16, PADB = 28;
  const plotH = H - PADT - PADB, plotW = W - PADL - PADR;
  const iw = plotW / months.length, bw = Math.min(36, iw * 0.62);
  const fill = { ideal: "#5cb56e", shoulder: "#e9a83a", off: "#a49bc4" };
  const inkc = { ideal: "#1c6b3c", shoulder: "#8a5b06", off: "#5b5480" };
  const grid = [0, 50, 100].map((g) => {
    const y = H - PADB - plotH * g / 100;
    return `<line x1="${PADL}" y1="${y}" x2="${W - PADR}" y2="${y}" stroke="var(--line-soft)" stroke-width="1"/>`
      + `<text x="${PADL - 7}" y="${y + 3.5}" text-anchor="end" font-size="10" font-weight="700" fill="var(--muted)">${g}%</text>`;
  }).join("");
  const bars = months.map((m, i) => {
    const p = seasonRainProb(m);
    const x = PADL + i * iw + (iw - bw) / 2;
    const h = Math.max(1.5, plotH * p / 100);
    const y = H - PADB - h;
    const best = s.best_month_num === m.m;
    return `<rect x="${x}" y="${y}" width="${bw}" height="${h}" rx="4" fill="${fill[m.band] || fill.off}" stroke="var(--ink)" stroke-width="1.5">`
      + `<title>${m.name}: rain on ~${m.rain_days} of ${SEASON_MONTH_DAYS[m.m - 1]} days (~${p}%) · ~${m.rain_mm} mm — ${m.band}</title></rect>`
      + `<text x="${x + bw / 2}" y="${y - 4}" text-anchor="middle" font-size="10" font-weight="800" fill="${inkc[m.band] || inkc.off}">${p}%</text>`
      + `<text x="${x + bw / 2}" y="${H - PADB + 15}" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--muted)">${m.name.slice(0, 3)}${best ? " ★" : ""}</text>`;
  }).join("");
  return `<div class="season-chart">
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Chance of rain by month for ${escapeHtml(s.country || "this place")}">
      ${grid}
      <line x1="${PADL}" y1="${H - PADB}" x2="${W - PADR}" y2="${H - PADB}" stroke="var(--ink)" stroke-width="2"/>
      ${bars}
    </svg>
    <p class="season-chart-note">🌧 <b>Chance of rain by month</b> — the share of days with ≥0.1 mm of rain in the 2023–2025 climate normals (Open-Meteo wet-day definition): ~40% ≈ rain on about 1 in 2–3 days. A climate average, not a forecast. Hover a bar for the numbers behind it.</p>
  </div>`;
}

/* Best time to visit — the key-free seasons dataset: month chips colored by band
   (green ideal / amber shoulder / gray off), the chance-of-rain chart, a
   12-month mini table, the month verdict when the user named one, holidays, and
   an honest note. Replaces the markdown section in place (the markdown is still
   emitted for CLI/SSE). */
function renderSeasons(s) {
  if (!s) return "";
  const country = s.country || "this place";
  if (!s.covered) {
    return `<section class="seasons-block">
      <h3>📅 Best time to visit — ${escapeHtml(country)}</h3>
      <p class="season-note">${escapeHtml(s.note || "I don't have verified seasonal data for this place yet.")}</p>
      <p class="note">I won't invent months — say the word and I'll research the best time on the live web instead (community-sourced — double-check before booking).</p>
    </section>`;
  }
  const chip = (m) => {
    const star = (s.best_month_num === m.m) ? " ★" : "";
    return `<span class="season-chip chip-${m.band}" title="${m.name}: high ${m.high}°C / low ${m.low}°C · ~${m.rain_mm} mm rain (${m.rain_days} wet days) — ${m.band}">${m.name.slice(0, 3)}${star}</span>`;
  };
  const chips = (s.months || []).map(chip).join("");
  const rows = (s.months || []).map((m) =>
    `<tr class="row-${m.band}"><td>${escapeHtml(m.name)}</td><td>${m.high}°</td><td>${m.low}°</td>` +
    `<td>~${m.rain_mm} mm</td><td>~${seasonRainProb(m)}%</td><td>${m.rain_days} d</td><td><b>${m.band}</b></td></tr>`).join("");
  const verdict = (s.month_verdict && s.month_verdict.verdict)
    ? `<p class="season-verdict verdict-${s.month_verdict.band}">📌 ${escapeHtml(s.month_verdict.verdict)}</p>` : "";
  const holidays = (s.holidays || []).length
    ? `<p class="season-holidays">🎉 <b>Public holidays:</b> ` +
      s.holidays.map((h) => escapeHtml(h.name + (h.date ? ` (${h.date})` : ""))).join(" · ") +
      (s.holidays_url ? ` <a class="season-hlink" href="${escapeHtml(s.holidays_url)}" target="_blank" rel="noopener">[source ↗]</a>` : "")
    : "";
  const prose = s.prose
    ? `<details class="season-prose"><summary>🌦️ Why these months (Wikivoyage climate notes, summarized)</summary>` +
      `<p>${escapeHtml(s.prose)}</p>${s.prose_url ? `<a href="${escapeHtml(s.prose_url)}" target="_blank" rel="noopener">[source ↗]</a>` : ""}</details>`
    : "";
  const ref = s.reference_point
    ? `<p class="season-ref">Climate reference point: <b>${escapeHtml(s.reference_point)}</b> (${escapeHtml(s.climate_years || "2023-2025")} normals)</p>` : "";
  return `<section class="seasons-block">
    <h3>📅 Best time to visit — ${escapeHtml(country)}</h3>
    <p class="seasons-summary">✨ <b>Ideal:</b> ${escapeHtml((s.ideal || []).join(", "))} &nbsp;·&nbsp; ` +
      `🌤 <b>Shoulder:</b> ${escapeHtml((s.shoulder || []).join(", "))} &nbsp;·&nbsp; ` +
      `🌧 <b>Off-season:</b> ${escapeHtml((s.off || []).join(", "))} &nbsp;·&nbsp; ` +
      `⭐ <b>Best overall:</b> ${escapeHtml(s.best_month || "")}</p>
    ${ref}${verdict}
    <div class="season-chips">${chips}</div>
    ${renderRainChart(s)}
    <table class="season-table">
      <thead><tr><th>Month</th><th>High</th><th>Low</th><th>Rain</th><th>Rain chance</th><th>Wet days</th><th>Band</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${holidays}${prose}
    <p class="note">Derived from REAL climate normals (Open-Meteo 2023-2025) + Wikipedia public holidays + Wikivoyage climate notes — key-free public sources, no API keys. Ideal/shoulder/off are a comfort+dryness estimate over those normals, not a forecast — events and local festivals can shift the real sweet spot.</p>
  </section>`;
}

/* At-a-glance trip calculator. It keeps assumptions visible: flight and lodging
   figures come from this response; local travel uses the rate and outing count
   the traveler chooses. */
/** "📖 {destination} in a minute" — the most important facts, shown ABOVE the Quick
    report so people learn the place fast. Grounded only (CP 1.1/3.1): gist from the
    KB, the can't-skip history from the KB (or a key-free Wikipedia intro for known
    countries), and the at-a-glance row from the world dataset + this run's
    seasons/safety. Any fact with no grounded source is omitted (never invented). */
function renderAreaBrief(ab) {
  if (!ab) return "";
  const facts = (ab.facts || []).filter((f) => f && (f.value));
  if (!ab.gist && !ab.history && !facts.length) return "";
  const factChips = facts.map((f) =>
    `<div class="ab-fact"><span class="ab-fact-label">${escapeHtml(f.label)}</span>` +
    `<span class="ab-fact-value">${renderInline(f.value)}</span></div>`).join("");
  const srcTag = ab.history_source ? ` <span class="ab-src">${escapeHtml(ab.history_source)}</span>` : "";
  return `<section class="area-brief">
    <div class="ab-head">
      <span class="ab-title">\uD83D\uDCD6 ${escapeHtml(ab.destination || "your trip")} in a minute</span>
      <span class="ab-tag">the facts worth knowing before you go</span>
    </div>
    ${ab.gist ? `<p class="ab-para"><span class="ab-kicker">What it is.</span> ${renderInline(ab.gist)}</p>` : ""}
    ${ab.history ? `<p class="ab-para"><span class="ab-kicker">The history you can't skip.</span> ${renderInline(ab.history)}${srcTag}</p>` : ""}
    ${factChips ? `<div class="ab-facts">${factChips}</div>` : ""}
    ${ab.provenance ? `<p class="ab-prov">${escapeHtml(ab.provenance)}</p>` : ""}
  </section>`;
}

/** Quick report (Do / Go / Stay) — the concise 'what most people recommend' rollup,
    shown at the TOP of the answer. Items may contain **bold** markdown (renderInline). */
function renderQuickReport(qr) {
  if (!qr) return "";
  const cols = [
    { key: "do",   title: "Do",   icon: "\uD83C\uDFE2" },
    { key: "go",   title: "Go",   icon: "\uD83D\uDCCD" },
    { key: "stay", title: "Stay", icon: "\uD83D\uDECF\uFE0F" },
  ];
  if (!cols.some((c) => (qr[c.key] || []).length)) return "";
  const col = (c) => {
    const items = (qr[c.key] || []).filter(Boolean);
    if (!items.length) return "";
    return `<div class="qr-col">
      <div class="qr-col-title"><span class="qr-icon">${c.icon}</span>${c.title}</div>
      <ul class="qr-list">${items.map((it) => `<li>${renderInline(it)}</li>`).join("")}</ul>
    </div>`;
  };
  return `<section class="quick-report">
    <div class="qr-head">
      <span class="qr-title">\uD83C\uDFAF GOALS \u2014 ${escapeHtml(qr.destination || "your trip")}</span>
      <span class="qr-tag">what most people recommend \u00B7 Do \u00B7 Go \u00B7 Stay</span>
    </div>
    <div class="qr-grid">${cols.map(col).join("")}</div>
    ${qr.provenance ? `<p class="qr-prov">${escapeHtml(qr.provenance)}</p>` : ""}
  </section>`;
}

/* Photos — landmark + dining/food scene photos from Wikimedia Commons (CC-licensed),
   each with caption + artist/license attribution (grounded, never invented). The
   markdown section is skipped in the sections loop (rendered here, richer). */
function renderPhotoFigure(im) {
  const cap = im.caption || im.title || "Photo";
  const credit = [im.artist, im.license].filter(Boolean).join(" · ");
  return `<figure class="photo-fig">
    <a class="photo-link" href="${escapeHtml(im.url)}" target="_blank" rel="noopener" title="Open original on Wikimedia Commons">
      <img src="/img?url=${encodeURIComponent(im.thumb_url)}" data-direct="${escapeHtml(im.thumb_url)}"
           onerror="this.onerror=null;this.src=this.dataset.direct;" alt="${escapeHtml(cap)}" loading="lazy" referrerpolicy="no-referrer">
    </a>
    <figcaption class="photo-cap">${escapeHtml(cap)}
      <span class="photo-credit">${escapeHtml(credit ? credit + " · " : "")}Wikimedia Commons</span>
    </figcaption>
  </figure>`;
}
function renderPhotos(photos) {
  if (!photos || !photos.length) return "";
  const groups = photos.map((g) =>
    `<div class="photo-group">
      <h4 class="photo-group-title">${escapeHtml(g.label)}</h4>
      <div class="photo-row">${(g.images || []).map(renderPhotoFigure).join("")}</div>
    </div>`).join("");
  return `<section class="photos-block">
    <h3>📸 Photos — landmarks & scenes</h3>
    ${groups}
    <p class="note">Photos from Wikimedia Commons (CC-licensed) with artist & license shown under each; click a photo for its original page. Shown only when a real, attributed image was found — none are generated or invented.</p>
  </section>`;
}

function renderTripSummary(s) {
  if (!s || (!s.flight_cost && !(s.hotels || []).length && !s.safety && !(s.places || []).length)) return "";
  const hotels = s.hotels || [];
  const places = s.places || [];
  const hotelOptions = hotels.length
    ? hotels.map((h, i) => `<option value="${i}">${escapeHtml(h.name)} — $${escapeHtml(h.price_usd)}/night${h.area ? ` · ${escapeHtml(h.area)}` : ""}</option>`).join("")
    : `<option value="">No priced stay retrieved</option>`;
  const placeChoices = places.length ? `<details class="summary-places"><summary>Choose places to include (${places.length} found)</summary>` +
    places.map((p, i) => `<label><input type="checkbox" class="summary-place" value="${i}"> ${escapeHtml(p.name)} <span>${escapeHtml(p.category)}${p.address ? ` · ${escapeHtml(p.address)}` : ""}</span></label>`).join("") + `</details>` : "";
  const safety = s.safety ? `<span class="summary-safety">🛡️ Level ${escapeHtml(s.safety.level)} · ${escapeHtml(s.safety.rating || s.safety.advisory || "Check advisory")}</span>` : `<span class="summary-safety muted">🛡️ Safety advisory unavailable</span>`;
  // 🛫 Flight details: supplements the cost figure with whatever was actually
  // retrieved (duration, nonstop/stops, airlines, fare band, times) — labeled
  // by source, never invented. Omitted entirely when there is no flight info.
  const fd = s.flight_details;
  let fdHtml = "";
  if (fd && (fd.routes || []).length) {
    const row = (r) => {
      const bits = [];
      const from = [r.origin, r.origin_code].filter(Boolean).join(" ");
      bits.push(`<span class="fd-route">🛫 ${escapeHtml(from || "—")} → ${escapeHtml(s.destination || "destination")}</span>`);
      if (r.direct) bits.push(`<span class="fd-tag fd-direct">nonstop</span>`);
      else if (r.stops) bits.push(`<span class="fd-tag">${escapeHtml(r.stops)}</span>`);
      if (r.duration) bits.push(`<span class="fd-meta">⏱ ${escapeHtml(r.duration)}</span>`);
      if (r.airlines && r.airlines.length) bits.push(`<span class="fd-meta">${escapeHtml(r.airlines.join(" · "))}</span>`);
      if (r.via_options && r.via_options.length) bits.push(`<span class="fd-meta">via ${escapeHtml(r.via_options.join(", "))}</span>`);
      if (r.flight_no) bits.push(`<span class="fd-meta">${escapeHtml(r.flight_no)}</span>`);
      if (r.times) bits.push(`<span class="fd-meta">${escapeHtml(r.times)}</span>`);
      if (r.date) bits.push(`<span class="fd-meta">${escapeHtml(r.date)}</span>`);
      if (r.price_usd) bits.push(`<span class="fd-price">$${r.price_usd}</span>`);
      if (r.economy_low && r.economy_high) bits.push(`<span class="fd-price">$${r.economy_low}–$${r.economy_high} band</span>`);
      return `<div class="fd-row">${bits.join("")}</div>`;
    };
    fdHtml = `<div class="flight-details">
      <div class="fd-head">Flight details <span class="fd-label">${escapeHtml(fd.label || "")}</span>${fd.url ? ` <a class="fd-src" href="${escapeHtml(fd.url)}" title="Open the route schedule source in your browser">source ↗</a>` : ""}</div>
      ${fd.routes.map(row).join("")}
    </div>`;
  }
  return `<section class="trip-summary" data-summary='${attrEscape(JSON.stringify(s))}'>
    <div class="summary-head"><div><h3>✦ Trip bottom line</h3><p>Choose a stay and local-travel assumptions to see the cost and distance before digging into details.</p></div>${safety}</div>
    <div class="summary-grid">
      <label>Stay<select class="summary-hotel">${hotelOptions}</select></label>
      <label>Nights<input class="summary-nights" type="number" min="1" value="${escapeHtml(s.nights || 3)}"></label>
      <label>Local outings<input class="summary-outings" type="number" min="0" value="${escapeHtml(s.nights || 3)}"></label>
      <label>Local transport $/mile<input class="summary-rate" type="number" min="0" step="0.01" value="0.67"></label>
    </div>
    ${placeChoices}
    ${fdHtml}
    <div class="summary-result" aria-live="polite"></div>
    <p class="note">Flight: ${escapeHtml(s.flight_cost ? `$${s.flight_cost} (${s.flight_note})` : s.flight_note || "not retrieved")}. Distance is an estimate: airport round trip + selected hotel-to-main-spots round trips. Local activities have no invented ticket prices.</p>
  </section>`;
}

function bindTripSummary(root) {
  const box = root.querySelector(".trip-summary");
  if (!box) return;
  let s = {}; try { s = JSON.parse(box.dataset.summary || "{}"); } catch { return; }
  const output = box.querySelector(".summary-result");
  if (!output) return;
  const recalc = () => {
    const hotel = (s.hotels || [])[Number(box.querySelector(".summary-hotel").value)] || {};
    const nights = Math.max(1, Number(box.querySelector(".summary-nights").value) || 1);
    const outings = Math.max(0, Number(box.querySelector(".summary-outings").value) || 0);
    const rate = Math.max(0, Number(box.querySelector(".summary-rate").value) || 0);
    const airport = Number(hotel.airport_mi) || 0;
    const main = Number(hotel.excursion_mi) || 0;
    const miles = airport * 2 + main * 2 * outings;
    const lodging = (Number(hotel.price_usd) || 0) * nights;
    const transport = miles * rate;
    const flight = Number(s.flight_cost) || 0;
    const total = flight + lodging + transport;
    const selected = box.querySelectorAll(".summary-place:checked").length;

    // Honest estimate: sum ONLY what we actually retrieved (lodging + local
    // travel, plus airfare when a real price was found). Whatever we did NOT
    // retrieve is listed plainly as "excluding …" so the number reads as a
    // floor, never a quote — we never invent airfare or local ticket prices.
    const excluding = [];
    if (!flight) excluding.push("flights / airfare");
    excluding.push("local dining, tickets & activities");

    output.innerHTML =
      `<strong>≈ $${total.toFixed(0)} estimated total</strong>` +
      `<span class="summary-break">$${lodging.toFixed(0)} lodging · $${transport.toFixed(0)} local travel · ${miles.toFixed(0)} local miles${selected ? ` · ${selected} place${selected === 1 ? "" : "s"}` : ""}</span>` +
      `<span class="summary-excl">Excluding: ${excluding.join(" · ")}</span>`;
  };
  box.querySelectorAll("select, input").forEach((node) => node.addEventListener("input", recalc));
  recalc();
}

function scoreBars(scores) {
  if (!scores) return "";
  const item = (label, key, total = false) => `
    <div class="score${total ? " total" : ""}">
      <div class="row"><span>${label}</span>
        <div class="bar"><span style="width:${Math.round((scores[key] ?? 0) * 100)}%"></span></div>
        <span class="val">${(scores[key] ?? 0).toFixed(3)}</span>
      </div>
    </div>`;
  return `<div class="scores">
    ${item("relevance", "relevance")}
    ${item("reliability", "reliability")}
    ${item("recency", "recency")}
    ${item("phrase", "phrase")}
    ${item("TOTAL", "total", true)}
  </div>`;
}

function sourceCard(src, n) {
  const text = escapeHtml(src.text || "");
  return `<article class="source-card" id="src-${n}">
    <div class="src-head">
      <span class="src-num">${n}</span>
      <div style="min-width:0">
        <div class="src-title">${escapeHtml(src.title || src.id || "source")}</div>
        <div class="src-meta">
          ${tierBadge(src.tier)}
          ${src.provider ? `<span class="src-provider" title="live-web search provider">⌗ ${escapeHtml(src.provider)}</span>` : ""}
          ${src.date ? `<span>${escapeHtml(src.date)}</span>` : ""}
          ${src.url ? `<a href="${escapeHtml(src.url)}" target="_blank" rel="noopener">${escapeHtml(src.url.replace(/^https?:\/\//, ""))} ↗</a>` : ""}
        </div>
      </div>
    </div>
    <p class="src-text clamped">${text}</p>
    <button class="src-expand" type="button" data-for="src-${n}">expand</button>
    ${scoreBars(src.scores)}
  </article>`;
}

function bindSourceClicks(root) {
  root.querySelectorAll(".cite").forEach((c) => {
    c.addEventListener("click", () => {
      const n = c.dataset.cite;
      const card = document.getElementById(`src-${n}`);
      if (!card) return;
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      card.classList.add("flash");
      setTimeout(() => card.classList.remove("flash"), 1600);
    });
  });
  root.querySelectorAll(".src-expand").forEach((b) => {
    b.addEventListener("click", () => {
      const card = document.getElementById(b.dataset.for);
      const p = card && card.querySelector(".src-text");
      if (!p) return;
      p.classList.toggle("clamped");
      b.textContent = p.classList.contains("clamped") ? "expand" : "collapse";
    });
  });
}

/* ================================================================
 * header: health / mode / kb stats
 * ================================================================ */
async function loadHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    const mode = $("#modeBadge");
    if (h.mode === "llm") {
      mode.textContent = `🧠 LLM planner · ${h.llm}`;
    } else {
      mode.textContent = "🧩 Simple planner (offline)";
    }
    const kb = h.kb || {};
    $("#kbStats").textContent = `📚 ${kb.documents ?? 0} guides · ${kb.chunks ?? 0} sections`;
  } catch {
    $("#modeBadge").textContent = "⚠️ server unreachable";
    $("#kbStats").textContent = "";
  }
}

/* ================================================================
 * tabs
 * ================================================================ */
function switchPanel(name) {
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("on", p.id === name));
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.panel === name;
    t.classList.toggle("on", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
  });
  if (name === "panelMemory") loadMemory();
}
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => switchPanel(t.dataset.panel)));

/* ================================================================
 * mode toggle + example chips
 * ================================================================ */
$("#modeAgent").addEventListener("click", () => setMode("agent"));
$("#modeDirect").addEventListener("click", () => setMode("direct"));
function setMode(m) {
  state.mode = m;
  $("#modeAgent").classList.toggle("on", m === "agent");
  $("#modeDirect").classList.toggle("on", m === "direct");
  $("#queryInput").placeholder = m === "agent"
    ? "e.g. Plan a 5 day trip to the Cayman Islands from Miami Sep 12 2026, budget $2500, with flights and where to stay"
    : "e.g. Kennywood discounts  (quick search of the guide library)";
}

/* ================================================================
 * question history — after the first inquiry the "Try:" examples
 * are replaced by your recent questions (newest first, click to
 * re-ask). Persisted in localStorage so it survives a page reload.
 * ================================================================ */
const HISTORY_KEY = "wayfinder.history.v1";
const HISTORY_MAX = 12;
const EXAMPLES_HTML = $("#exampleChips").innerHTML; // captured once, before any re-render
let history = (() => {
  try {
    const raw = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(raw) ? raw.filter((x) => typeof x === "string" && x.trim()) : [];
  } catch { return []; }
})();

function saveHistory() {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch { /* storage unavailable */ }
}

function renderChipArea() {
  const box = $("#exampleChips");
  if (history.length) {
    const items = history.map((q) =>
      `<button type="button" class="chip hist" data-q="${escapeHtml(q)}" title="${escapeHtml(q)}">🕘 ${escapeHtml(q)}</button>`
    ).join("");
    box.innerHTML =
      `<span class="chips-label">Recent:</span>${items}` +
      `<button type="button" class="chip ghost" data-act="examples" title="Show the example prompts">✦ examples</button>` +
      `<button type="button" class="chip ghost" data-act="clear" title="Clear question history">🗑 clear</button>`;
  } else {
    box.innerHTML = EXAMPLES_HTML;
  }
}

function addHistory(q) {
  history = [q, ...history.filter((x) => x !== q)].slice(0, HISTORY_MAX);
  saveHistory();
  renderChipArea();
}

function clearHistory() {
  history = [];
  saveHistory();
  renderChipArea();
}

function renderExamples() { $("#exampleChips").innerHTML = EXAMPLES_HTML; }

// If the browser has no local history yet, seed the "Recent:" chips from the
// server-side request log (GET /api/requests) — the single source of truth —
// so history survives a fresh browser / cleared storage. LocalStorage wins if present.
async function seedHistoryFromServer() {
  if (history.length) return; // local history already exists — do not overwrite
  try {
    const r = await fetch("/api/requests?n=" + HISTORY_MAX);
    const data = await r.json();
    const seeded = [];
    const seen = new Set();
    for (const req of data.requests || []) {
      if (req.kind !== "agent") continue; // only real agent questions become chips
      const q = String(req.query || "").trim();
      if (!q || seen.has(q)) continue;
      seen.add(q);
      seeded.push(q);
      if (seeded.length >= HISTORY_MAX) break;
    }
    if (seeded.length) {
      history = seeded; // newest first (server returns newest first)
      saveHistory();
      renderChipArea();
    }
  } catch { /* server offline — keep the example prompts */ }
}

$("#exampleChips").addEventListener("click", (e) => {
  const btn = e.target.closest("button.chip");
  if (!btn) return;
  if (btn.dataset.act === "clear") { clearHistory(); return; }
  if (btn.dataset.act === "examples") { renderExamples(); return; }
  $("#queryInput").value = btn.dataset.q;
  submit();
});

$("#queryForm").addEventListener("submit", (e) => { e.preventDefault(); submit(); });
$("#queryInput").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); submit(); } });
$("#clearTrace").addEventListener("click", () => {
  $("#traceList").hidden = true;
  $("#traceList").innerHTML = "";
  $("#traceEmpty").hidden = false;
  state.traceCount = 0;
  $("#badgeTrace").hidden = true;
});

/* ================================================================
 * Home base — "📍 Fly from": preselect a country (and optionally a
 * region/state) so flight answers anchor on YOUR gateway — Nevada ->
 * Las Vegas (LAS), Turkey -> Istanbul (IST) — instead of random US
 * hubs. Dropdown data: GET /api/airports (47-country offline dataset,
 * data/airports.json). Persisted in localStorage; sent with every
 * /api/query as {country, region}. An explicit "from X" in the query
 * always wins over the home base (enforced server-side).
 * ================================================================ */
const HOME_KEY = "wf_home";
let homeCountries = [];            // [{name, regions: [...]}] from /api/airports
let homeSel = { country: "", region: "" };

function loadHome() {
  try {
    const raw = JSON.parse(localStorage.getItem(HOME_KEY) || "null");
    if (raw && typeof raw.country === "string" && raw.country) {
      homeSel = { country: raw.country, region: typeof raw.region === "string" ? raw.region : "" };
    }
  } catch { /* storage unavailable */ }
}

function saveHome() {
  try {
    if (homeSel.country) localStorage.setItem(HOME_KEY, JSON.stringify(homeSel));
    else localStorage.removeItem(HOME_KEY);
  } catch { /* storage unavailable */ }
}

function currentHome() {
  return homeSel.country ? { country: homeSel.country, region: homeSel.region || "" } : null;
}

function renderHomeBar() {
  const cs = $("#homeCountry");
  const rs = $("#homeRegion");
  if (!cs || !rs) return;
  cs.innerHTML = '<option value="">—</option>' + homeCountries
    .map((c) => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`).join("");
  const entry = homeCountries.find((c) => c.name === homeSel.country) || null;
  if (entry && entry.regions.length) {
    rs.innerHTML = '<option value="">whole country</option>' + entry.regions
      .map((r) => `<option value="${escapeHtml(r)}">${escapeHtml(r)}</option>`).join("");
    rs.disabled = false;
  } else {
    rs.innerHTML = '<option value="">whole country</option>';
    rs.disabled = true;
  }
  cs.value = homeSel.country;
  rs.value = homeSel.region || "";
  const chip = $("#homeChip");
  if (homeSel.country) {
    chip.textContent = homeSel.region
      ? `📍 From: ${homeSel.region}, ${homeSel.country}  ✕`
      : `📍 From: ${homeSel.country}  ✕`;
    chip.hidden = false;
  } else {
    chip.hidden = true;
  }
}

async function initHomeBar() {
  loadHome();
  try {
    const resp = await fetch("/api/airports");
    if (resp.ok) {
      const data = await resp.json();
      if (Array.isArray(data.countries)) homeCountries = data.countries;
    }
  } catch { /* endpoint unavailable: bar stays inert, requests go home-less */ }
  // drop a persisted selection the dataset no longer knows (honesty over stale UI)
  if (homeSel.country && !homeCountries.some((c) => c.name === homeSel.country)) {
    homeSel = { country: "", region: "" };
    saveHome();
  }
  renderHomeBar();
  $("#homeCountry").addEventListener("change", (e) => {
    homeSel.country = e.target.value;
    homeSel.region = "";
    saveHome();
    renderHomeBar();
  });
  $("#homeRegion").addEventListener("change", (e) => {
    homeSel.region = e.target.value;
    saveHome();
    renderHomeBar();
  });
  $("#homeChip").addEventListener("click", () => {
    homeSel = { country: "", region: "" };
    saveHome();
    renderHomeBar();
  });
}

/* ================================================================
 * SSE: POST /api/query via fetch + ReadableStream
 * ================================================================ */
async function streamQuery(query, mode, signal, home) {
  const resp = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode, home: home || null }),
    signal,
  });
  if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of frame.split("\n")) {
        const l = line.trim();
        if (!l.startsWith("data:")) continue;
        try { handleEvent(JSON.parse(l.slice(5).trim())); } catch { /* partial/garbage frame: skip */ }
      }
    }
  }
}

/* ================================================================
 * event handling (agent trace + answer)
 * ================================================================ */
const EV_META = {
  memory:          { icon: "🗂️",  label: "memory" },
  thought:         { icon: "🧠",  label: "thought" },
  action:          { icon: "🔍",  label: "action" },
  observation:     { icon: "📥",  label: "observation" },
  prune:           { icon: "✂️",  label: "trim" },
  answer:          { icon: "💬",  label: "answer" },
  guardrail:       { icon: "🛡️",  label: "safety check" },
  cancelled:       { icon: "⏹️",  label: "cancelled" },
  done:            { icon: "✅",  label: "done" },
  search_results:  { icon: "🔎",  label: "results" },
};

function traceSticky() {
  const list = $("#traceList");
  const dist = list.scrollHeight - list.scrollTop - list.clientHeight;
  return dist < 120;
}

function appendTrace(ev) {
  $("#traceEmpty").hidden = true;
  const list = $("#traceList");
  list.hidden = false;
  const meta = EV_META[ev.type] || { icon: "•", label: ev.type };
  const sticky = traceSticky();
  let body = "";
  if (ev.type === "thought") body = `<div>${escapeHtml(ev.text)}</div>`;
  else if (ev.type === "memory") body = `<div>${escapeHtml(ev.note)}</div>`;
  else if (ev.type === "action") body =
    `<div><span class="tool">${escapeHtml(ev.tool)}</span><span class="muted">tool call</span></div>
     <pre>${escapeHtml(JSON.stringify(ev.input, null, 2))}</pre>`;
  else if (ev.type === "observation") body = `<div>${escapeHtml(ev.observation)}</div>`;
  else if (ev.type === "prune") body =
    `<div>${ev.title ? `<b>${escapeHtml(ev.title)}</b> — score ${ev.score ?? "?"}` : "trim"}
     <div class="prune-reason">${escapeHtml(ev.reason || "")}</div></div>`;
  else if (ev.type === "answer") body = `<div>Answer composed — ${escapeHtml((ev.intro || "").slice(0, 140))}…</div>`;
  else if (ev.type === "guardrail") {
    if (ev.checks) {
      body = `<div class="gr-summary">` + Object.entries(ev.checks).map(([k, v]) =>
        `<div><b>${escapeHtml(k)}:</b> ${escapeHtml(v)}</div>`).join("") + `</div>`;
    } else {
      body = `<div><span class="gr-check">${escapeHtml(ev.check || ev.layer || "safety check")}</span> ` +
             `<b class="gr-result">${escapeHtml(ev.result || "")}</b> ` +
             (ev.detail ? `<span class="muted">— ${escapeHtml(ev.detail)}</span>` : "") + `</div>`;
    }
  }
  else if (ev.type === "cancelled") body = `<div>${escapeHtml(ev.note || "Stream cancelled by the user")}</div>`;
  else if (ev.type === "done") body =
    `<div>steps=${ev.steps ?? "?"} · searches=${ev.searches ?? "?"} · trimmed=${ev.pruned ?? "?"} · ${ev.elapsed_ms ?? "?"} ms · mode=${escapeHtml(ev.mode || "?")}` +
    (typeof ev.confidence === "number" ? ` · 🛡️ ${ev.confidence}% ${escapeHtml(ev.confidence_label || "")}` : "") + `</div>`;
  else if (ev.type === "search_results") body =
    `<div>${(ev.results || []).length} results kept, ${(ev.puned || []).length} set aside (shown in the Answer tab)</div>`;

  const stepTag = ev.step ? `<span class="ev-step">step ${ev.step}</span>` : "";
  list.appendChild(el(
    `<li class="ev ev-${ev.type}">
       <div class="ev-head">
         <span class="ev-icon">${meta.icon}</span>
         <span class="ev-type">${meta.label}</span>
         ${stepTag}
         <span class="ev-ts">${escapeHtml(ev.ts || "")}</span>
       </div>
       <div class="ev-body">${body || escapeHtml(JSON.stringify(ev))}</div>
     </li>`));
  if (sticky) list.scrollTop = list.scrollHeight;

  state.traceCount++;
  const badge = $("#badgeTrace");
  badge.textContent = state.traceCount;
  badge.hidden = false;
}

function handleEvent(ev) {
  if (ev.type === "answer") { renderAnswer(ev); appendTrace(ev); return; }
  if (ev.type === "search_results") { renderDirectResults(ev); appendTrace(ev); return; }
  if (ev.type === "cancelled") { setRunStatus(null); appendTrace(ev); return; }
  if (ev.type === "done") { setRunStatus(null); appendTrace(ev); return; }
  appendTrace(ev);
}

/* ================================================================
 * BOOK IT — front-and-center booking links (flights / stays / tours)
 * Opens the big booking sites with the destination (and dates, when we
 * have them) pre-filled. We never auto-book — the user compares & decides.
 * ================================================================ */
function qs(pairs) {
  const s = pairs.filter((p) => p[1] != null && p[1] !== "")
    .map((p) => `${p[0]}=${encodeURIComponent(p[1])}`).join("&");
  return s ? ("?" + s) : "";
}
function addDays(iso, n) {
  try {
    const d = new Date(iso + "T00:00:00Z");
    d.setUTCDate(d.getUTCDate() + Number(n));
    return d.toISOString().slice(0, 10);
  } catch { return null; }
}
function renderBooking(goal) {
  const dest = (goal.destination || "").trim();
  if (!dest) return "";
  const enc = encodeURIComponent(dest);
  const slug = dest.toLowerCase().trim().replace(/\s+/g, "-").replace(/[^\w-]/g, "");
  const iso = /^\d{4}-\d{2}-\d{2}$/.test(goal.date || "") ? goal.date : null;
  const co = (iso && Number(goal.nights)) ? addDays(iso, Number(goal.nights)) : null;
  const origin = goal.origin || "";
  const budget = goal.budget ? Number(goal.budget) : null;

  const flightQ = (origin ? "flights from " + origin + " to " + dest : "flights to " + dest) + (iso ? " on " + iso : "");
  const flights = [
    { name: "Google Flights", url: "https://www.google.com/travel/flights" + qs([["q", flightQ]]) },
    { name: "Kayak", url: slug ? "https://www.kayak.com/destination/" + slug : "https://www.kayak.com/" },
  ];
  const stays = [
    { name: "Booking.com", url: "https://www.booking.com/searchresults.html" + qs([["ss", dest], ["checkin", iso], ["checkout", co], ["no_rooms", "1"], ["group_adults", "2"]]) },
    { name: "Airbnb", url: "https://www.airbnb.com/s/" + enc + "/homes" + qs([["check_in", iso], ["check_out", co]]) },
  ];
  const tours = [
    { name: "GetYourGuide", url: "https://www.getyourguide.com/s" + qs([["q", dest]]) },
    { name: "Klook", url: "https://www.klook.com/en-US/search/" + qs([["q", dest]]) },
  ];

  const col = (icon, label, links, hint) =>
    `<div class="bk-col">
       <div class="bk-col-head">${icon} ${label}</div>
       <div class="bk-links">
         ${links.map(l => `<a class="bk-link" href="${l.url}" target="_blank" rel="noopener" aria-label="Open ${escapeHtml(l.name)} in a new tab">${escapeHtml(l.name)} <span class="bk-arr">↗</span></a>`).join("")}
       </div>
       ${hint ? `<div class="bk-hint">${hint}</div>` : ""}
     </div>`;

  const when = iso ? `dates ${iso}${co ? " → " + co : ""}` : "your dates";
  return `
  <div class="bk-panel" role="region" aria-label="Book ${escapeHtml(dest)}">
    <div class="bk-head">
      <span class="bk-title">✈️ Book it — ${escapeHtml(dest)}${budget ? " · ~$" + budget.toLocaleString() + " budget" : ""}</span>
      <span class="bk-sub">Tap a site to compare prices & book — pre-filled for ${escapeHtml(dest)} (${when}).</span>
    </div>
    <div class="bk-grid">
      ${col("✈️", "Flights", flights, origin ? "From " + origin + " · pick your flights" : "Pick your route & dates")}
      ${col("🏨", "Stays", stays, iso ? "Showing " + iso + " → " + (co || "…") : "Pick your dates on the site")}
      ${col("🎟️", "Tours & Experiences", tours, "Day trips, tours & tickets")}
    </div>
  </div>`;
}

/* ================================================================
 * answer rendering
 * ================================================================ */
function renderAnswer(ev) {
  state.answerSources = ev.sources || [];
  const parts = [];

  // BOOK IT — front and center: direct links to book flights, stays, tours.
  if (ev.goal && (ev.goal.destination || "")) {
    parts.push(renderBooking(ev.goal));
  }

  if (ev.intro) parts.push(`<p class="answer-intro">${renderInline(ev.intro)}</p>`);

  // Area brief ("in a minute") — the most important facts, ABOVE the Quick report so
  // people learn the place fast. Its markdown section is skipped in the loop below
  // (rendered here, richer) — same in-place pattern as seasons/leads.
  const AB_HEADER = "## \uD83D\uDCD6 ";
  if (ev.area_brief) {
    parts.push(renderAreaBrief(ev.area_brief));
  }

  // Quick report (Do / Go / Stay) — the concise 'what most people recommend' rollup,
  // shown at the TOP of the answer. Its markdown section is skipped in the loop below
  // (rendered here, richer) — same in-place pattern as seasons/leads.
  const QUICK_HEADER = "## \uD83C\uDFAF GOALS";
  if (ev.quick_report && (ev.quick_report.do || ev.quick_report.go || ev.quick_report.stay)) {
    parts.push(renderQuickReport(ev.quick_report));
  }

  // Photos (landmarks + dining/food scenes) — right after the Quick report card.
  // Its markdown section is skipped in the loop below (rendered here, richer).
  const PHOTOS_HEADER = "## \uD83D\uDCF8 Photos";
  if (ev.photos && ev.photos.length) {
    parts.push(renderPhotos(ev.photos));
  }

  // CP 6.1 L4 — output scoring: the confidence band is shown for reassurance.
  if (typeof ev.confidence_pct === "number") {
    const cls = (ev.confidence_label || "medium").toLowerCase();
    parts.push(`<div class="confidence conf-${cls}">
      <span class="conf-badge">🛡️ ${ev.confidence_pct}% · ${escapeHtml(ev.confidence_label || "")}</span>
      ${ev.confidence_detail ? `<span class="conf-detail">${escapeHtml(ev.confidence_detail)}</span>` : ""}
    </div>`);
  }

  if (ev.goal && Object.values(ev.goal).some((v) => v)) {
    const g = ev.goal;
    const chip = (k, v) => v ? `<span class="goal-chip"><b>${escapeHtml(k)}:</b> ${escapeHtml(Array.isArray(v) ? v.join(", ") : v)}</span>` : "";
    parts.push(`<div class="answer-goal">
      ${chip("destination", g.destination)}${chip("origin", g.origin)}${chip("date", g.date)}
      ${g.budget ? chip("budget", "$" + g.budget) : ""}${chip("nights", g.nights)}
      ${chip("focus", g.focus)}
    </div>`);
  }

  if (ev.trip_summary) parts.push(renderTripSummary(ev.trip_summary));

  // Cross-source research leads: render as per-source dropdowns (link + quote + note).
  // The markdown bullet section is still emitted for CLI/SSE text consumers; the UI
  // replaces it in place with the richer dropdown block.
  const leads = (ev.leads || []).filter((l) => l && (l.title || l.url));
  const LEADS_HEADER = "## 🗂️ Cross-source research leads";
  // Best time to visit: month chips + mini table + holidays (structured `seasons`
  // field) replace the markdown section in place — same pattern as the leads.
  const seasons = ev.seasons && (ev.seasons.covered || ev.seasons.note) ? ev.seasons : null;
  const SEASONS_HEADER = "## 📅 Best time to visit";
  for (const s of ev.sections || []) {
    const head = String(s).trimStart();
    if (ev.area_brief && head.startsWith(AB_HEADER)) {
      continue;  // already rendered above the Quick report as renderAreaBrief (richer)
    }
    if (head.startsWith(QUICK_HEADER)) {
      continue;  // already rendered at the top as renderQuickReport (richer)
    }
    if (head.startsWith(PHOTOS_HEADER)) {
      continue;  // already rendered as renderPhotos right after the Quick report
    }
    if (leads.length && head.startsWith(LEADS_HEADER)) {
      parts.push(renderLeads(leads));
      continue;
    }
    if (seasons && head.startsWith(SEASONS_HEADER)) {
      parts.push(renderSeasons(seasons));
      continue;
    }
    parts.push(`<section class="answer-section">${renderSection(s)}</section>`);
  }
  if (leads.length && !(ev.sections || []).some((s) => String(s).trimStart().startsWith(LEADS_HEADER))) {
    parts.push(renderLeads(leads));  // fallback if the section text is missing
  }
  if (seasons && !(ev.sections || []).some((s) => String(s).trimStart().startsWith(SEASONS_HEADER))) {
    parts.push(renderSeasons(seasons));  // fallback if the section text is missing
  }

  if ((ev.sources || []).length) {
    parts.push(`<h3 class="sources-title">Sources</h3>`);
    ev.sources.forEach((src, i) => parts.push(sourceCard(src, i + 1)));
  }

  if ((ev.followups || []).length) {
    parts.push(`<div class="followups"><span class="followups-label">Try asking next:</span>` +
      ev.followups.map((f) => `<button type="button" class="follow-chip" data-q="${escapeHtml(f)}">${escapeHtml(f)}</button>`).join("") +
      `</div>`);
  }

  $("#answerContent").innerHTML = parts.join("");
  bindSourceClicks($("#answerContent"));
  $("#answerContent").querySelectorAll(".follow-chip").forEach((b) =>
    b.addEventListener("click", () => { $("#queryInput").value = b.dataset.q; submit(); }));
  bindTripSummary($("#answerContent"));
  setRunStatus(null);
  const badge = $("#badgeAnswer");
  badge.textContent = "1";
  badge.hidden = false;
  window.scrollTo({ top: $("#panelAnswer").offsetTop - 70, behavior: "smooth" });
}

function renderDirectResults(ev) {
  const parts = [];
  parts.push(`<p class="answer-intro">Quick search — <b>${(ev.results || []).length}</b> results kept,
     <b>${(ev.puned || []).length}</b> set aside (ranked by relevance and trust).</p>`);
  (ev.results || []).forEach((src, i) => parts.push(sourceCard(src, i + 1)));
  if ((ev.puned || []).length) {
    parts.push(`<h3 class="sources-title">Set aside (with reasons)</h3>` +
      `<div class="prune-list">` +
      ev.puned.map((p) =>
        `<div class="prune-item"><b>✂️ ${escapeHtml(p.title || p.id)}</b> <span>score ${p.score}</span> <span class="muted">${escapeHtml(p.reason || "")}</span></div>`
      ).join("") + `</div>`);
  }
  $("#answerContent").innerHTML = parts.join("");
  bindSourceClicks($("#answerContent"));
  setRunStatus(null);
  $("#badgeAnswer").textContent = "1";
  $("#badgeAnswer").hidden = false;
}

function setRunStatus(text) {
  const box = $("#runStatus");
  if (!text) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  box.innerHTML = `<span class="spinner"></span><span>${escapeHtml(text)}</span>`;
}

/* ================================================================
 * submit
 * ================================================================ */
async function submit() {
  const q = $("#queryInput").value.trim();
  if (!q || state.running) return;
  addHistory(q);  // newest first — the chip row swaps to "Recent:" after this

  switchPanel("panelAnswer");
  $("#answerEmpty").hidden = true;
  $("#answerBody").hidden = false;
  $("#answerContent").innerHTML = "";
  setRunStatus(state.mode === "agent" ? "Wayfinder is thinking — steps are streaming into the Step-by-step tab…"
                                      : "Running a quick search of the guide library…");
  setRunning(true);
  state.controller = new AbortController();

  try {
    await streamQuery(q, state.mode, state.controller.signal, currentHome());
  } catch (err) {
    if (err && err.name === "AbortError") {
      $("#answerContent").innerHTML =
        `<div class="answer-section"><p class="note">⏹️ Run cancelled — you stay in control ` +
        `(you keep control; safety checks run automatically).</p></div>`;
    } else {
      $("#answerContent").innerHTML =
        `<div class="answer-section"><p class="note">⚠️ Request failed: ${escapeHtml(err.message)}. Is the server running? (python app.py)</p></div>`;
    }
  } finally {
    state.controller = null;
    setRunning(false);
  }
}

$("#stopBtn").addEventListener("click", () => {
  if (state.controller) state.controller.abort();
});

function setRunning(on) {
  state.running = on;
  $("#goBtn").disabled = on;
  $("#stopBtn").hidden = !on;
  $("#runDot").hidden = !on;
}

/* ================================================================
 * guide library tab — browsable list with client-side filter
 * ================================================================ */
let kbAllDocs = [];  // cached so the filter re-renders instantly

async function loadKb() {
  try {
    const r = await fetch("/api/kb");
    const kb = await r.json();
    kbAllDocs = kb.doc_list || [];
    renderKbDocs("");
  } catch {
    $("#kbDocs").innerHTML = `<div class="mem-none">Could not load the guide library — is the server running?</div>`;
  }
}

function renderKbDocs(filter) {
  const q = (filter || "").toLowerCase().trim();
  const docs = q
    ? kbAllDocs.filter((d) => {
        const hay = `${d.title || d.id} ${d.category || ""} ${d.source || ""} ${d.tier || ""}`.toLowerCase();
        return hay.includes(q);
      })
    : kbAllDocs;

  if (!docs.length) {
    $("#kbDocs").innerHTML = `<div class="mem-none">No guides match “${escapeHtml(filter)}”.</div>`;
    return;
  }

  // group by category
  const groups = {};
  docs.forEach((d) => {
    const cat = d.category || "general";
    (groups[cat] = groups[cat] || []).push(d);
  });

  const catIcons = {
    general: "🧭", country_guide: "🌍", dining: "🍽️", hotels: "🏨",
    shopping: "🛍️", attractions: "🎢", safety: "🛡️", transit: "🚌",
    seasons: "📅", food: "🍽️", culture: "🎭", nature: "🌿",
    nightlife: "🌙", gifts: "🎁", overview: "📋", best_time: "📅",
  };

  const sortedCats = Object.keys(groups).sort((a, b) => a.localeCompare(b));
  const html = sortedCats.map((cat) => {
    const icon = catIcons[cat] || "📄";
    const items = groups[cat].map((d) => {
      // extract destination from title (strip "— World Travel Guide", "Best Time to Visit", etc.)
      let dest = (d.title || d.id || "").split(/\s+—\s+|\s+Best Time\s/i)[0].trim();
      // remove generic suffixes
      dest = dest.replace(/\s*\(.*?\)/g, "").trim();
      return `
      <div class="doc-card kb-plan-card" data-dest="${escapeHtml(dest)}">
        <div class="kb-plan-head">
          <h4>${escapeHtml(d.title || d.id)}</h4>
          <button type="button" class="plan-btn" data-dest="${escapeHtml(dest)}" title="Pull up a full plan for this destination">✦ Plan this</button>
        </div>
        <div class="doc-meta">
          ${tierBadge(d.tier)}
          ${d.source ? `<span>${escapeHtml(d.source)}</span>` : ""}
          ${d.date ? `<span>${escapeHtml(d.date)}</span>` : ""}
          ${d.url ? `<a href="${escapeHtml(d.url)}" target="_blank" rel="noopener">source ↗</a>` : ""}
        </div>
      </div>`;
    }).join("");
    return `<div class="kb-group"><h3 class="kb-group-title">${icon} ${escapeHtml(cat.charAt(0).toUpperCase() + cat.slice(1))} <span class="muted">${groups[cat].length}</span></h3><div class="kb-group-items">${items}</div></div>`;
  }).join("");

  $("#kbDocs").innerHTML = html;
}

$("#kbFilter").addEventListener("input", (e) => renderKbDocs(e.target.value));

/* "Plan this" — click a card to launch a full plan for that destination */
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".plan-btn");
  if (!btn) return;
  const dest = btn.dataset.dest || btn.closest("[data-dest]")?.dataset.dest || "";
  if (!dest) return;
  const budget = $("#kbBudget").value;
  const days = $("#kbDays").value || "5";
  let q = `Plan a ${days} day trip to ${dest} with flights and where to stay`;
  if (budget) q += `, budget $${parseInt(budget).toLocaleString()}`;
  // switch to Full plan mode + submit
  setMode("agent");
  switchPanel("panelAnswer");
  $("#queryInput").value = q;
  submit();
});

/* ================================================================
 * memory tab
 * ================================================================ */
async function loadMemory() {
  try {
    const r = await fetch("/api/memory");
    const m = await r.json();

    // bookings
    const bookings = m.bookings || [];
    $("#memBookings").innerHTML = bookings.length ? bookings.map((b) => {
      const f = b.flight || {};
      return `<div class="mem-card">
        <div class="mem-head"><b>${escapeHtml(b.ref || "booking")}</b> <span class="mem-ts">${escapeHtml(b.status || "")}</span></div>
        <div class="mem-sub">${escapeHtml(f.airline || "")} ${escapeHtml(f.flight_no || "")}
          · ${escapeHtml(f.date || "")} ${escapeHtml(f.dep_time || "")}→${escapeHtml(f.arr_time || "")}
          · $${escapeHtml(f.price_usd ?? "?")} · passenger: ${escapeHtml(b.passenger || "?")}</div>
      </div>`;
    }).join("") : `<div class="mem-none">No bookings yet. Ask Wayfinder to book a flight — it stays pending until you confirm.</div>`;

    // episodic
    const ep = m.episodic || [];
    $("#memEpisodic").innerHTML = ep.length ? ep.map((r) => `
      <div class="mem-card">
        <div class="mem-head"><b>“${escapeHtml(r.query)}”</b> <span class="mem-ts">${escapeHtml(r.ts || "")}</span></div>
        ${r.destination ? `<div class="mem-sub">destination: ${escapeHtml(r.destination)} · cache key: <code>${escapeHtml(r.cache_key || "")}</code></div>` : ""}
        <div class="mem-actions">${(r.actions || []).map((a) => `<span class="tool-chip">${escapeHtml(a.tool)}</span>`).join("")}</div>
        ${r.top_sources && r.top_sources.length ? `<div class="mem-sub">top sources: ${r.top_sources.map(escapeHtml).join(" · ")}</div>` : ""}
        ${r.summary ? `<div class="mem-sub">→ ${escapeHtml(r.summary)}</div>` : ""}
      </div>`).join("")
      : `<div class="mem-none">No activity yet — it appears after your first search.</div>`;

    // semantic
    const sem = m.semantic || [];
    $("#memSemantic").innerHTML = sem.length ? sem.map((s) => `
      <div class="mem-card">
        <div class="mem-head"><span class="mem-role ${s.role === "user" ? "user" : "agent"}">${s.role === "user" ? "🧑 you" : "🤖 Wayfinder"}</span></div>
        <div class="mem-sub">${escapeHtml(s.text)}</div>
      </div>`).join("")
      : `<div class="mem-none">No recent context yet.</div>`;
  } catch {
    $("#memBookings").innerHTML = `<div class="mem-none">Could not load /api/memory — is the server running?</div>`;
    $("#memEpisodic").innerHTML = "";
    $("#memSemantic").innerHTML = "";
  }

  // request log — separate fetch so a failure here can't hide the memory above.
  // data/request_log.jsonl (server-side) is the single source of truth for
  // what the user asked; this panel makes that tracking visible.
  try {
    const rr = await fetch("/api/requests?n=20");
    const rd = await rr.json();
    const reqs = rd.requests || [];
    $("#memRequests").innerHTML = reqs.length ? reqs.map((r) => {
      const conf = (r.confidence_pct != null)
        ? ` · ${r.confidence_pct}% ${escapeHtml(r.confidence_label || "")}`
        : "";
      const dest = r.destination ? ` · ${escapeHtml(r.destination)}` : "";
      const kept = (r.kind === "search") ? ` · ${r.kept ?? "?"} kept / ${r.pruned ?? "?"} set aside` : "";
      return `<div class="mem-card">
        <div class="mem-head">
          <b>“${escapeHtml(r.query || "(no query)")}”</b>
          <span class="mem-ts"><span class="req-mode req-${escapeHtml(r.kind || "search")}">${escapeHtml(r.kind || "search")}</span> ${escapeHtml(r.ts || "")}</span>
        </div>
        <div class="mem-sub">status: ${escapeHtml(r.status || "ok")}${dest}${kept}${(r.elapsed_ms != null) ? ` · ${r.elapsed_ms}ms` : ""}${conf}</div>
      </div>`;
    }).join("")
      : `<div class="mem-none">No logged requests yet — every query and search lands here (GET /api/requests, newest first).</div>`;
    if (rd.total != null) {
      const extra = rd.total - reqs.length;
      if (extra > 0) $("#memRequests").insertAdjacentHTML("beforeend", `<div class="mem-none">+ ${extra} older request${extra === 1 ? "" : "s"} in the log.</div>`);
    }
  } catch {
    $("#memRequests").innerHTML = `<div class="mem-none">Could not load /api/requests — is the server running?</div>`;
  }
}
$("#refreshMemory").addEventListener("click", loadMemory);

/* ================================================================
 * init
 * ================================================================ */
// optional URL controls (handy for demos & screenshots):
//   /?demo=kennywood | cayman | book | safety   auto-submits a sample query
//   /?tab=answer | trace | kb | memory          opens that panel
const DEMOS = {
  cayman: "Plan a 5 day trip to the Cayman Islands from Miami Sep 12 2026, budget $2500, with flights and where to stay",
  kennywood: "What discounts can I get for Kennywood in Pittsburgh and where should I stay?",
  book: "Book me a flight to the Cayman Islands from Orlando",
  safety: "Where is safe to stay in George Town and how do I get around?",
};
const TABS = { answer: "panelAnswer", trace: "panelTrace", kb: "panelKb", memory: "panelMemory" };

/* ================================================================
 * In-app link viewer — clicking any http(s) link opens it in an
 * embedded frame (stay in Wayfinder) with an "Open in browser"
 * escape hatch for sites that refuse framing (Reddit, Instagram and
 * many retailers send X-Frame-Options / CSP frame-ancestors).
 * Wired via ONE delegated handler so every rendered link (source
 * cards, research leads, season sources, photo figures, KB docs)
 * routes through it — no need to touch each renderer.
 * ================================================================ */
const lv = { url: "", timer: null, onLoaded: null };

// Open a URL in the user's browser using the most webview-friendly mechanism
// available. A synthetic <a target="_blank"> click on user gesture works where
// window.open() popups are blocked; top-level navigation is the last resort.
function openExternal(url, tab) {
  if (!url) return;
  try {
    const a = document.createElement("a");
    a.href = url;
    a.target = tab ? "_self" : "_blank";
    a.rel = "noopener noreferrer";
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (err) { /* ignore */ }
}

function copyUrl(url) {
  if (!url) return;
  const done = () => { const b = $("#lvCopy"); if (b) { b.textContent = "✓ Copied"; setTimeout(() => { b.textContent = "⧉ Copy"; }, 1400); } };
  if (navigator.clipboard && navigator.clipboard.writeText) { navigator.clipboard.writeText(url).then(done).catch(done); }
  else { const t = document.createElement("textarea"); t.value = url; t.style.position = "fixed"; t.style.opacity = "0"; document.body.appendChild(t); t.select(); try { document.execCommand("copy"); } catch (e) {} t.remove(); done(); }
}

function openLinkViewer(url, label) {
  if (!url) return;
  if (!$("#linkViewer")) { openExternal(url); return; }        // defensive: markup missing
  if (!/^https?:\/\//i.test(url)) { openExternal(url); return; } // mailto:, tel: -> native
  lv.url = url;
  const v = $("#linkViewer"), frame = $("#lvFrame"), blocked = $("#lvBlocked");
  $("#lvUrlText").textContent = (label || url).replace(/^https?:\/\//i, "");
  $("#lvBlockedMsg").textContent =
    "Still blank after a moment? Some sites (Reddit, Instagram, many retailers) block in-app " +
    "display. Use “Open in browser” below — or copy the URL and paste it into your browser.";
  if (lv.onLoaded) frame.removeEventListener("load", lv.onLoaded);
  lv.onLoaded = () => { clearTimeout(lv.timer); blocked.hidden = true; };
  frame.addEventListener("load", lv.onLoaded);
  blocked.hidden = true;
  frame.hidden = false;
  v.hidden = false;
  document.body.classList.add("lv-open");
  frame.src = url;  // navigate now that the overlay is in layout
  clearTimeout(lv.timer);
  lv.timer = setTimeout(() => { blocked.hidden = false; }, 8000);  // no load -> likely blocked
  $("#lvClose").focus();
}

function closeLinkViewer() {
  $("#linkViewer").hidden = true;
  document.body.classList.remove("lv-open");
  clearTimeout(lv.timer);
  const frame = $("#lvFrame");
  if (lv.onLoaded) { frame.removeEventListener("load", lv.onLoaded); lv.onLoaded = null; }
  frame.src = "about:blank";
}

// One delegated handler for every external link in the app.
//  • Default click  -> open in your browser (new tab) — the reliable path.
/* ---------- photo lightbox: click a photo to blow it up ---------- */
function openPhotoViewer(link) {
  const v = $("#photoViewer");
  const fig = link && link.closest ? link.closest(".photo-fig") : null;
  const img = fig ? fig.querySelector("img") : null;
  if (!v || !fig || !img) return;
  const cap = fig.querySelector(".photo-cap");
  const cred = cap ? cap.querySelector(".photo-credit") : null;
  const credit = cred ? cred.textContent.trim() : "";
  const caption = cap ? cap.textContent.replace(credit, "").trim() : (img.alt || "Photo");
  const srcImg = $("#pvImg");
  srcImg.alt = caption;
  srcImg.src = img.currentSrc || img.src;      // the actually-rendered image (incl. fallback)
  $("#pvCaption").textContent = caption;
  const credEl = $("#pvCredit");
  credEl.textContent = credit;
  credEl.hidden = !credit;
  v.dataset.original = link.href || "";
  v.hidden = false;
  document.body.classList.add("pv-open");
  $("#pvClose").focus();
}
function closePhotoViewer() {
  const v = $("#photoViewer");
  if (!v) return;
  v.hidden = true;
  document.body.classList.remove("pv-open");
}

//  • click a photo -> blow it up in the lightbox
//  • Alt/Option+click -> open in the in-app viewer (for where framing is allowed).
document.addEventListener("click", (e) => {
  const a = (e.target && e.target.closest) ? e.target.closest("a[href]") : null;
  if (!a || a.closest("#linkViewer")) return;          // viewer's own controls stay native
  const href = a.getAttribute("href") || "";
  if (!/^https?:\/\//i.test(href) || e.button !== 0) return;
  if (a.classList.contains("photo-link") && !e.altKey && !e.metaKey && !e.ctrlKey && !e.shiftKey) {
    e.preventDefault(); openPhotoViewer(a); return;
  }
  if (e.altKey) { e.preventDefault(); openLinkViewer(href, (a.textContent || "").trim()); return; }
  if (e.metaKey || e.ctrlKey || e.shiftKey) return;    // let the browser handle modifier new-tab natively
  e.preventDefault();
  openExternal(href);                                   // default -> open in browser (new tab)
});

document.addEventListener("keydown", (e) => {
  const pv = $("#photoViewer");
  if (pv && !pv.hidden && e.key === "Escape") { closePhotoViewer(); return; }
  const v = $("#linkViewer");
  if (v && !v.hidden && e.key === "Escape") closeLinkViewer();
});

// Attach viewer controls (guarded so a stale cached page can't crash the app).
if (typeof document !== "undefined" && document.querySelector) {
  const _open = $("#lvOpen"); if (_open) _open.addEventListener("click", () => openExternal(lv.url));
  const _blockedOpen = $("#lvBlockedOpen"); if (_blockedOpen) _blockedOpen.addEventListener("click", () => openExternal(lv.url));
  const _selfOpen = $("#lvSelf"); if (_selfOpen) _selfOpen.addEventListener("click", () => { if (lv.url) location.href = lv.url; });
  const _copy = $("#lvCopy"); if (_copy) _copy.addEventListener("click", () => copyUrl(lv.url));
  const _close = $("#lvClose"); if (_close) _close.addEventListener("click", closeLinkViewer);
  const _pvOpen = $("#pvOpen"); if (_pvOpen) _pvOpen.addEventListener("click", () => {
    const v = $("#photoViewer"); if (v && v.dataset.original) openExternal(v.dataset.original);
  });
  const _pvClose = $("#pvClose"); if (_pvClose) _pvClose.addEventListener("click", closePhotoViewer);
  const _pvStage = $("#pvStage"); if (_pvStage) _pvStage.addEventListener("click", (e) => {
    if (e.target === _pvStage) closePhotoViewer();
  });
}

(async function init() {
  await seedHistoryFromServer();  // fresh browser? seed chips from the server request log
  renderChipArea();  // show persisted question history (or the examples, if none)
  await initHomeBar();  // 📍 Fly from — home base dropdowns (47-country dataset)
  await loadHealth();
  loadKb();
  loadMemory();
  const params = new URLSearchParams(location.search);
  const tab = TABS[params.get("tab")];
  if (tab) switchPanel(tab);
  const demo = DEMOS[params.get("demo")];
  if (demo) {
    $("#queryInput").value = demo;
    setTimeout(async () => {
      await submit();
      if (tab && tab !== "panelAnswer") switchPanel(tab); // e.g. view the trace after the run
    }, 350);
    return;
  }
  $("#queryInput").focus();
})();
