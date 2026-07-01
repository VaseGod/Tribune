"use strict";

/* ------------------------------------------------------------------ *
 * Small helpers
 * ------------------------------------------------------------------ */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function money(n) {
  if (n === null || n === undefined || n === "") return "—";
  const v = Number(n);
  return "$" + v.toLocaleString(undefined, { maximumFractionDigits: 0 });
}
function titleCase(s) {
  return String(s).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
async function api(method, url, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}
function showLoader(text) {
  $("#loader-text").textContent = text || "Running the verification pipeline…";
  $("#loader").hidden = false;
}
function hideLoader() { $("#loader").hidden = true; }
let toastTimer;
function toast(msg, isErr) {
  const t = $("#toast");
  t.textContent = msg; t.className = "toast" + (isErr ? " err" : ""); t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 4200);
}

/* ------------------------------------------------------------------ *
 * State
 * ------------------------------------------------------------------ */
const State = {
  meta: null,
  demoCases: [],
  activeRun: null,           // { case, result, audit_chain_verified, program_names }
  chat: [],                  // {role, content}
  settings: { provider: "local_rules", jurisdiction: "EX", abstention_threshold: 0.7 },
  view: "demo",
  wizardStep: 0,
};

const VIEW_META = {
  demo: ["Demo Cases", "Run a realistic, fully-simulated case through the verification pipeline."],
  wizard: ["New Case", "Enter a situation and intake documents — TRIBUNE assesses, cites, and abstains when unsure."],
  settings: ["Settings", "Choose the model backend, jurisdiction, and how cautious TRIBUNE should be."],
  results: ["Assessment", "Cited, verifiable results. Anything binding requires your explicit sign-off."],
};

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */
async function boot() {
  try {
    const saved = JSON.parse(localStorage.getItem("tribune.settings") || "null");
    if (saved) State.settings = { ...State.settings, ...saved };
  } catch (_) {}

  bindChrome();
  try {
    State.meta = await api("GET", "/api/meta");
    State.settings.jurisdiction = State.settings.jurisdiction || State.meta.defaults.jurisdiction;
    State.settings.abstention_threshold = State.meta.defaults.abstention_threshold;
  } catch (e) {
    toast("Could not reach the TRIBUNE API: " + e.message, true);
  }
  setView("demo");
}

function bindChrome() {
  $$("#nav .nav-item").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  $("#chat-launcher").addEventListener("click", toggleChat);
  $("#chat-close").addEventListener("click", () => { $("#chat-drawer").hidden = true; });
  $("#chat-form").addEventListener("submit", onChatSubmit);
  $("#audit-toggle").addEventListener("change", (e) => {
    const w = $("#audit-wrap"); if (w) w.hidden = !e.target.checked;
  });
}

function setView(name) {
  if (name !== "results") State.view = name;
  $$("#nav .nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === (name === "results" ? "demo" : name)));
  const [title, sub] = VIEW_META[name];
  $("#view-title").textContent = title;
  $("#view-sub").textContent = sub;
  $("#audit-toggle-wrap").hidden = name !== "results";

  const view = $("#view");
  view.style.animation = "none"; void view.offsetWidth; view.style.animation = "";
  if (name === "demo") renderDemo(view);
  else if (name === "wizard") renderWizard(view);
  else if (name === "settings") renderSettings(view);
  else if (name === "results") renderResults(view);
}

/* ------------------------------------------------------------------ *
 * Demo view
 * ------------------------------------------------------------------ */
async function renderDemo(view) {
  view.innerHTML = `<div class="grid" id="demo-grid"><div class="empty">Loading demo cases…</div></div>`;
  try {
    if (!State.demoCases.length) State.demoCases = await api("GET", "/api/cases/demo");
  } catch (e) {
    $("#demo-grid").innerHTML = `<div class="empty">Could not load demo cases: ${esc(e.message)}</div>`;
    return;
  }
  const grid = $("#demo-grid");
  grid.innerHTML = State.demoCases.map(demoCard).join("");
  $$(".js-run-demo", grid).forEach((b) =>
    b.addEventListener("click", () => runCase({ case_id: b.dataset.case }))
  );
}

function demoCard(c) {
  const s = c.situation;
  const names = c.program_names.join(" · ");
  return `<div class="card interactive">
    <div class="card-head">
      <h3>${esc(prettyCaseId(c.case_id))}</h3>
      <span class="chip">${esc(c.jurisdiction)}</span>
    </div>
    <div class="facts">
      <div class="fact"><b>Household</b>${s.household_size}</div>
      <div class="fact"><b>Monthly income</b>${money(s.monthly_income)}</div>
      <div class="fact"><b>Citizenship</b>${titleCase(s.citizenship_status)}</div>
      <div class="fact"><b>Employment</b>${titleCase(s.employment_status)}</div>
    </div>
    <div class="tags">${c.program_names.map((n) => `<span class="tag">${esc(n)}</span>`).join("")}</div>
    <button class="btn primary block js-run-demo" data-case="${esc(c.case_id)}">Run assessment</button>
  </div>`;
}

function prettyCaseId(id) {
  return titleCase(id.replace(/^demo-/, "").replace(/-/g, " "));
}

/* ------------------------------------------------------------------ *
 * Run a case
 * ------------------------------------------------------------------ */
async function runCase(extra) {
  const payload = {
    provider: State.settings.provider,
    abstention_threshold: State.settings.abstention_threshold,
    ...extra,
  };
  showLoader();
  try {
    const data = await api("POST", "/api/cases/run", payload);
    State.activeRun = data;
    State.chat = [];
    hideLoader();
    $("#audit-toggle").checked = false;
    setView("results");
    maybeSeedChat();
  } catch (e) {
    hideLoader();
    toast("Run failed: " + e.message, true);
  }
}

/* ------------------------------------------------------------------ *
 * Results view
 * ------------------------------------------------------------------ */
function renderResults(view) {
  const run = State.activeRun;
  if (!run) { view.innerHTML = `<div class="empty">No active assessment. Run a demo or new case.</div>`; return; }
  const names = run.program_names || {};
  const c = run.case;
  const r = run.result;

  view.innerHTML = `
    <div class="case-banner">
      <div>
        <h2 style="margin:0 0 6px;font-size:20px;">${esc(prettyCaseId(c.case_id))}</h2>
        <div class="meta">
          <span>Jurisdiction <b>${esc(c.jurisdiction)}</b></span>
          <span>Programs <b>${c.target_programs.length}</b></span>
          <span>Provider <b>${esc(run.settings.provider)}</b></span>
          <span>Abstention threshold <b>${run.settings.abstention_threshold.toFixed(2)}</b></span>
          <span>Audit chain ${run.audit_chain_verified ? '<b class="badge-ok">verified ✓</b>' : '<b class="badge-bad">broken ✗</b>'}</span>
        </div>
      </div>
      <button class="btn ghost sm" id="back-btn">← New / demo</button>
    </div>
    <div class="program-stack">${r.outcomes.map((o) => programCard(o, names)).join("")}</div>
    ${auditView(r.audit)}
  `;

  $("#back-btn").addEventListener("click", () => setView(State.view === "results" ? "demo" : State.view));
  bindAccordions(view);
  bindSubmit(view);
  const audit = $("#audit-wrap"); if (audit) audit.hidden = !$("#audit-toggle").checked;
}

function statusOf(o) {
  if (o.abstained) return { key: "amber", label: "Abstained · routed to a human" };
  const s = o.assessment ? o.assessment.status : null;
  if (s === "likely_eligible") return { key: "emerald", label: "Likely eligible" };
  if (s === "likely_ineligible") return { key: "crimson", label: "Likely ineligible" };
  return { key: "amber", label: "Indeterminate" };
}
const GAUGE_COL = { emerald: "#10b981", amber: "#f59e0b", crimson: "#ef4444" };

function programCard(o, names) {
  const st = statusOf(o);
  const name = names[o.program] || titleCase(o.program);
  const conf = o.abstention ? o.abstention.calibrated_confidence
    : (o.assessment ? o.assessment.self_confidence : 0);
  const pct = Math.round((conf || 0) * 100);

  const citMap = {};
  if (o.assessment) (o.assessment.citations || []).forEach((c) => { citMap[c.citation_id] = c; });

  let reason = "";
  if (o.abstained && o.abstention) {
    reason = `<div class="reason-banner amber">
      <strong>Why TRIBUNE held back:</strong> ${esc(o.abstention.reason)}.
      <div class="human-cta"><button class="btn sm js-ask">Ask the navigator about this →</button></div>
    </div>`;
  } else if (o.assessment && o.assessment.status === "likely_ineligible") {
    reason = `<div class="reason-banner crimson">
      This does not appear to meet the rules. This is not final — a navigator can confirm or help you contest it.
      <div class="human-cta"><button class="btn sm js-ask">Discuss options →</button></div>
    </div>`;
  }

  const crits = o.assessment ? o.assessment.criteria.map((cr) => critRow(cr, citMap)).join("") : "";
  const waitlist = (o.assessment && o.assessment.waitlist_status)
    ? `<div class="notice">🏚 Housing note — <b>eligibility is not access</b>: waitlist status is <b>${esc(o.assessment.waitlist_status)}</b>.</div>` : "";

  return `<div class="card program-card s-${st.key}" data-program="${esc(o.program)}">
    <div class="pc-head">
      <div class="pc-title">
        <h3>${esc(name)}</h3>
        <span class="status-pill">${esc(st.label)}</span>
        ${o.replans ? `<span class="muted tiny">${o.replans} replan(s) before a verified result</span>` : ""}
      </div>
      <div class="gauge" style="--val:${pct};--col:${GAUGE_COL[st.key]}">
        <span class="g-val">${pct}%</span>
        <span class="g-lbl">${o.abstention ? "calibrated" : "confidence"}</span>
      </div>
    </div>
    ${reason}
    ${waitlist}
    ${crits ? `<div class="accordion">${crits}</div>` : ""}
    ${o.materials ? preparer(o) : ""}
  </div>`;
}

function critRow(cr, citMap) {
  const map = { satisfied: ["ok", "✓"], not_satisfied: ["no", "✗"], unknown: ["unk", "?"] };
  const [cls, ico] = map[cr.outcome] || ["unk", "?"];
  const cites = (cr.citation_ids || []).map((id) => citMap[id]).filter(Boolean).map((c) => {
    const isUrl = /^https?:/.test(c.locator || "");
    const src = isUrl
      ? `<a class="src" href="${esc(c.locator)}" target="_blank" rel="noopener">${esc(c.source)}</a>`
      : `<span class="src">${esc(c.source)}</span>`;
    return `<div class="cite">${src}<span>${esc(c.title)}</span></div>`;
  }).join("");
  return `<div class="acc-row">
    <button class="acc-head" type="button">
      <span class="crit-ico ${cls}">${ico}</span>
      <span>${esc(cr.description)}</span>
      <span class="acc-caret">▸</span>
    </button>
    <div class="acc-body"><div class="acc-inner">
      <div>Outcome: <b>${titleCase(cr.outcome)}</b></div>
      ${cites ? `<div class="cites">${cites}</div>` : `<div class="muted tiny" style="margin-top:6px">No citation needed (criterion was not resolved).</div>`}
    </div></div>
  </div>`;
}

function preparer(o) {
  const m = o.materials;
  const checklist = (m.document_checklist || []).map((d) => `<li>${esc(d)}</li>`).join("");
  const fieldEntries = Object.entries(m.application_fields || {}).slice(0, 8);
  const fields = fieldEntries.map(([k, v]) => `<div class="k">${esc(titleCase(k))}</div><div>${esc(v)}</div>`).join("");
  const appeal = m.appeal_packet
    ? `<div class="notice">Appeal packet drafted — deadline window ${esc(m.appeal_packet.filing_window_days)} days, est. ${esc(m.appeal_packet.estimated_days_remaining)} days remaining.</div>`
    : "";
  return `<div class="preparer">
    <h4>📋 Prepared materials <span class="muted tiny">(nothing submitted)</span></h4>
    <ul class="checklist">${checklist}</ul>
    ${fieldEntries.length ? `<div class="muted tiny" style="margin-bottom:6px">Drafted application fields</div><div class="fields-preview">${fields}</div>` : ""}
    ${appeal}
    <p class="notice">${esc(m.action_required_note)}</p>
    <div class="js-submit-zone">
      <button class="btn primary js-review">Review &amp; authorize submission</button>
    </div>
  </div>`;
}

function bindAccordions(root) {
  $$(".acc-head", root).forEach((h) =>
    h.addEventListener("click", () => h.closest(".acc-row").classList.toggle("open"))
  );
  $$(".js-ask", root).forEach((b) => b.addEventListener("click", () => { openChat(); }));
}

/* ------------------------------------------------------------------ *
 * Action-gate submission flow
 * ------------------------------------------------------------------ */
function bindSubmit(root) {
  $$(".js-review", root).forEach((btn) => {
    btn.addEventListener("click", () => {
      const card = btn.closest(".program-card");
      const program = card.dataset.program;
      const zone = btn.closest(".js-submit-zone");
      zone.innerHTML = `
        <div class="field full" style="margin-top:6px">
          <label class="lbl">Authorizing navigator / caseworker name</label>
          <input type="text" class="js-signer" placeholder="e.g., Navigator Jane Doe" />
        </div>
        <div class="switch-row" style="margin-top:10px">
          <button class="btn primary js-confirm" data-program="${esc(program)}">Sign &amp; authorize</button>
          <button class="btn ghost js-cancel">Cancel</button>
        </div>`;
      $(".js-confirm", zone).addEventListener("click", () => confirmSubmit(zone, program));
      $(".js-cancel", zone).addEventListener("click", () => { zone.innerHTML = `<button class="btn primary js-review">Review &amp; authorize submission</button>`; bindSubmit(zone); });
    });
  });
}

async function confirmSubmit(zone, program) {
  const name = $(".js-signer", zone).value.trim();
  if (!name) { toast("Enter the authorizing person's name.", true); return; }
  const btn = $(".js-confirm", zone); btn.disabled = true; btn.textContent = "Signing…";
  try {
    const res = await api("POST", "/api/cases/submit-action", {
      case_id: State.activeRun.case.case_id, program, authorized_by: name,
    });
    const r = res.receipt;
    zone.innerHTML = `<div class="receipt">
      <h4>✓ Submission authorized by a human</h4>
      <div>Authorized by <b>${esc(r.authorized_by)}</b> for <b>${titleCase(r.program)}</b> at ${esc(r.authorized_at)}.</div>
      <div class="mono">sign-off token: ${esc(r.signoff_token)}</div>
      <div class="notice" style="margin-bottom:0">This signed receipt is the binding action — TRIBUNE itself never submitted anything.</div>
    </div>`;
    toast("Submission authorized and recorded.");
  } catch (e) {
    btn.disabled = false; btn.textContent = "Sign & authorize";
    toast("Authorization failed: " + e.message, true);
  }
}

/* ------------------------------------------------------------------ *
 * Audit log
 * ------------------------------------------------------------------ */
function auditView(records) {
  const rows = (records || []).map((rec) => `
    <div class="audit-row">
      <div class="audit-seq">#${rec.sequence}</div>
      <div class="state-tag">${esc(rec.state.toUpperCase())}</div>
      <div>
        <div class="audit-action"><b>${esc(rec.agent)}</b> — ${esc(rec.action)}</div>
        <div class="audit-hash">model: ${esc(rec.model_name)} · hash: ${esc((rec.record_hash || "").slice(0, 24))}… · prev: ${esc((rec.prev_hash || "∅").slice(0, 16))}…</div>
      </div>
    </div>`).join("");
  return `<div class="audit-wrap" id="audit-wrap" hidden>
    <h3>🔗 Hash-chained audit log <span class="muted tiny">(${(records || []).length} records — every step is recorded and tamper-evident)</span></h3>
    ${rows}
  </div>`;
}

/* ------------------------------------------------------------------ *
 * Wizard
 * ------------------------------------------------------------------ */
const WIZARD_STEPS = ["Household", "Income & status", "Programs & documents"];

function renderWizard(view) {
  State.wizardStep = 0;
  view.innerHTML = `<div class="wizard">
    <div class="stepper" id="stepper"></div>
    <form id="wizard-form">
      <div class="wstep" data-step="0">${stepHousehold()}</div>
      <div class="wstep" data-step="1" hidden>${stepIncomeStatus()}</div>
      <div class="wstep" data-step="2" hidden>${stepPrograms()}</div>
    </form>
    <div class="wizard-actions">
      <button class="btn ghost" id="wiz-back" disabled>← Back</button>
      <button class="btn primary" id="wiz-next">Next →</button>
    </div>
  </div>`;
  renderStepper();
  $("#wiz-back").addEventListener("click", () => gotoStep(State.wizardStep - 1));
  $("#wiz-next").addEventListener("click", onWizardNext);
  // employment toggle reveals separation fields
  const emp = $("#f-employment_status");
  if (emp) emp.addEventListener("change", () => {
    $("#unemp-fields").hidden = emp.value !== "unemployed";
  });
}

function renderStepper() {
  $("#stepper").innerHTML = WIZARD_STEPS.map((label, i) => {
    const cls = i === State.wizardStep ? "active" : (i < State.wizardStep ? "done" : "");
    const line = i < WIZARD_STEPS.length - 1 ? `<div class="step-line"></div>` : "";
    return `<div class="step ${cls}"><span class="num">${i < State.wizardStep ? "✓" : i + 1}</span>${esc(label)}</div>${line}`;
  }).join("");
}

function gotoStep(n) {
  n = Math.max(0, Math.min(WIZARD_STEPS.length - 1, n));
  State.wizardStep = n;
  $$(".wstep").forEach((s) => { s.hidden = Number(s.dataset.step) !== n; });
  renderStepper();
  $("#wiz-back").disabled = n === 0;
  $("#wiz-next").textContent = n === WIZARD_STEPS.length - 1 ? "Run assessment →" : "Next →";
}

function onWizardNext() {
  if (State.wizardStep < WIZARD_STEPS.length - 1) { gotoStep(State.wizardStep + 1); return; }
  submitWizard();
}

function stepHousehold() {
  return `<h3 class="section-title">Your household</h3>
  <p class="hint">Only what's needed to check the rules. This is simulated data — no real personal information.</p>
  <div class="form-grid">
    <div class="field"><label class="lbl">Household size</label><input type="number" id="f-household_size" min="1" value="2" /></div>
    <div class="field"><label class="lbl">Applicant age</label><input type="number" id="f-age" min="0" value="34" /></div>
    <div class="field full"><div class="switch-row">
      <label class="switch"><input type="checkbox" id="f-has_dependent_child" /> Has a dependent child</label>
      <label class="switch"><input type="checkbox" id="f-pregnant" /> Pregnant</label>
      <label class="switch"><input type="checkbox" id="f-disabled" /> Has a disability</label>
    </div></div>
  </div>`;
}

function stepIncomeStatus() {
  const jurisdictions = (State.meta?.jurisdictions || ["EX", "NX"]).map((j) => `<option value="${j}">${j}</option>`).join("");
  return `<h3 class="section-title">Income, status & location</h3>
  <div class="form-grid">
    <div class="field"><label class="lbl">Monthly income ($)</label><input type="number" id="f-monthly_income" min="0" value="1500" /></div>
    <div class="field"><label class="lbl">Liquid assets ($)</label><input type="number" id="f-liquid_assets" min="0" value="500" /></div>
    <div class="field"><label class="lbl">Monthly rent ($)</label><input type="number" id="f-monthly_rent" min="0" value="900" /></div>
    <div class="field"><label class="lbl">Jurisdiction</label><select id="f-jurisdiction">${jurisdictions}</select></div>
    <div class="field"><label class="lbl">Citizenship / immigration</label>
      <select id="f-citizenship_status">
        <option value="citizen">U.S. citizen</option>
        <option value="qualified_immigrant">Qualified immigrant</option>
        <option value="undocumented">Other / undocumented</option>
      </select></div>
    <div class="field"><label class="lbl">Employment</label>
      <select id="f-employment_status"><option value="employed">Employed</option><option value="unemployed">Unemployed</option></select></div>
    <div class="field full"><div class="switch-row">
      <label class="switch"><input type="checkbox" id="f-resident" checked /> Resident of this jurisdiction</label>
    </div></div>
  </div>
  <fieldset class="fieldset" id="unemp-fields" hidden>
    <legend>Unemployment details</legend>
    <div class="form-grid">
      <div class="field"><label class="lbl">Reason for separation</label>
        <select id="f-separation_reason">
          <option value="">—</option>
          <option value="laid_off">Laid off</option>
          <option value="fired_misconduct">Fired (misconduct)</option>
          <option value="quit_no_cause">Quit (no good cause)</option>
          <option value="quit_good_cause">Quit (with good cause)</option>
        </select></div>
      <div class="field"><label class="lbl">Base-period earnings ($)</label><input type="number" id="f-base_period_earnings" min="0" value="0" /></div>
      <div class="field"><label class="lbl">Weeks worked</label><input type="number" id="f-weeks_worked" min="0" value="0" /></div>
    </div>
  </fieldset>
  <fieldset class="fieldset">
    <legend>Housing & appeals (optional)</legend>
    <div class="form-grid">
      <div class="field"><label class="lbl">Waitlist status</label>
        <select id="f-waitlist_status"><option value="unknown">Unknown</option><option value="open">Open</option><option value="closed">Closed</option></select></div>
      <div class="field"><label class="lbl">Days since denial (appeals)</label><input type="number" id="f-days_since_denial" min="0" placeholder="leave blank if N/A" /></div>
      <div class="field full"><label class="lbl">Appeal grounds (appeals)</label><input type="text" id="f-appeal_grounds" placeholder="why you believe a denial was wrong" /></div>
    </div>
  </fieldset>`;
}

function stepPrograms() {
  const progs = State.meta?.benefit_programs || ["snap", "unemployment", "medicaid", "housing"];
  const allProgs = State.meta?.programs || progs.concat(["appeals"]);
  const names = State.meta?.program_names || {};
  const checks = allProgs.map((p) => `
    <label class="prog-check"><input type="checkbox" class="f-program" value="${p}" ${progs.includes(p) ? "checked" : ""}/> ${esc(names[p] || titleCase(p))}</label>`).join("");
  return `<h3 class="section-title">Programs & intake documents</h3>
  <p class="hint">Choose which programs to assess, and paste any intake document text (pay stub, benefit letter, denial notice). Use <code>key: value</code> lines to add evidence (e.g. <code>monthly_income: 1500</code>).</p>
  <div class="field full"><label class="lbl">Programs to assess</label><div class="prog-checks">${checks}</div></div>
  <div class="field full" style="margin-top:16px"><label class="lbl">Intake documents (simulated)</label>
    <textarea id="f-document_text" placeholder="monthly_income: 1500&#10;household_size: 2&#10;resident: true&#10;citizenship_status: citizen"></textarea></div>`;
}

function numOrNull(id) { const v = $("#" + id)?.value; return v === "" || v == null ? null : Number(v); }

function submitWizard() {
  const overrides = {
    household_size: numOrNull("f-household_size"),
    age: numOrNull("f-age"),
    has_dependent_child: $("#f-has_dependent_child").checked,
    pregnant: $("#f-pregnant").checked,
    disabled: $("#f-disabled").checked,
    monthly_income: numOrNull("f-monthly_income"),
    liquid_assets: numOrNull("f-liquid_assets"),
    monthly_rent: numOrNull("f-monthly_rent"),
    citizenship_status: $("#f-citizenship_status").value,
    employment_status: $("#f-employment_status").value,
    resident: $("#f-resident").checked,
    waitlist_status: $("#f-waitlist_status").value,
  };
  const sep = $("#f-separation_reason")?.value;
  if (sep) overrides.separation_reason = sep;
  const bpe = numOrNull("f-base_period_earnings"); if (bpe) overrides.base_period_earnings = bpe;
  const ww = numOrNull("f-weeks_worked"); if (ww) overrides.weeks_worked = ww;
  const dsd = numOrNull("f-days_since_denial"); if (dsd !== null) overrides.days_since_denial = dsd;
  const ag = $("#f-appeal_grounds")?.value.trim(); if (ag) overrides.appeal_grounds = ag;

  const target_programs = $$(".f-program").filter((c) => c.checked).map((c) => c.value);
  if (!target_programs.length) { toast("Choose at least one program.", true); return; }
  const document_text = $("#f-document_text").value.trim() || null;

  runCase({
    jurisdiction: $("#f-jurisdiction").value,
    overrides,
    target_programs,
    document_text,
  });
}

/* ------------------------------------------------------------------ *
 * Settings
 * ------------------------------------------------------------------ */
function renderSettings(view) {
  const s = State.settings;
  const jurisdictions = (State.meta?.jurisdictions || ["EX", "NX"]).map((j) =>
    `<option value="${j}" ${j === s.jurisdiction ? "selected" : ""}>${j}</option>`).join("");
  view.innerHTML = `<div class="settings">
    <div class="card">
      <h3 style="margin-top:0">Model backend</h3>
      <p class="muted tiny" style="margin-top:0">TRIBUNE is model-agnostic. The deterministic local engine runs offline; a self-hosted model plugs in via an OpenAI-compatible endpoint.</p>
      <div class="radio-row">
        <div class="radio-card ${s.provider === "local_rules" ? "sel" : ""}" data-provider="local_rules">
          <div class="rc-title">Local rules engine</div><div class="rc-desc">Deterministic, offline, reproducible. Default.</div>
        </div>
        <div class="radio-card ${s.provider === "openai_compat" ? "sel" : ""}" data-provider="openai_compat">
          <div class="rc-title">Self-hosted model</div><div class="rc-desc">vLLM / SGLang endpoint. If unreachable, TRIBUNE safely abstains.</div>
        </div>
      </div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">Jurisdiction</h3>
      <div class="field"><select id="set-jurisdiction">${jurisdictions}</select></div>
      <p class="muted tiny">EX = example expansion state · NX = example non-expansion state (Medicaid coverage gap).</p>
    </div>
    <div class="card">
      <h3 style="margin-top:0">Abstention threshold</h3>
      <p class="muted tiny" style="margin-top:0">Below this calibrated confidence, TRIBUNE abstains and routes to a human. Higher = more cautious.</p>
      <div class="range-row">
        <input type="range" id="set-threshold" min="0.5" max="0.95" step="0.01" value="${s.abstention_threshold}" />
        <span class="range-val" id="set-threshold-val">${s.abstention_threshold.toFixed(2)}</span>
      </div>
    </div>
  </div>`;

  $$(".radio-card").forEach((c) => c.addEventListener("click", () => {
    State.settings.provider = c.dataset.provider;
    $$(".radio-card").forEach((x) => x.classList.toggle("sel", x === c));
    persistSettings();
  }));
  $("#set-jurisdiction").addEventListener("change", (e) => { State.settings.jurisdiction = e.target.value; persistSettings(); });
  const range = $("#set-threshold");
  range.addEventListener("input", (e) => {
    State.settings.abstention_threshold = Number(e.target.value);
    $("#set-threshold-val").textContent = Number(e.target.value).toFixed(2);
  });
  range.addEventListener("change", persistSettings);
}
function persistSettings() {
  localStorage.setItem("tribune.settings", JSON.stringify(State.settings));
  toast("Settings saved — applied to your next run.");
}

/* ------------------------------------------------------------------ *
 * Chat
 * ------------------------------------------------------------------ */
function toggleChat() { const d = $("#chat-drawer"); d.hidden ? openChat() : (d.hidden = true); }
function openChat() {
  $("#chat-drawer").hidden = false;
  if (!State.chat.length) maybeSeedChat(true);
  renderChat();
  $("#chat-text").focus();
}
function maybeSeedChat(force) {
  if (State.chat.length) return;
  let greeting = "Hi — I can explain why TRIBUNE reached its conclusions, or tell you what documents would help. Ask me anything.";
  if (State.activeRun) {
    const abst = State.activeRun.result.outcomes.filter((o) => o.abstained).length;
    if (abst) greeting = `I see TRIBUNE routed ${abst} program${abst > 1 ? "s" : ""} to a human. Ask me “why?” or “what's missing?” and I'll explain — grounded in your actual case.`;
  }
  State.chat.push({ role: "assistant", content: greeting });
  if (force) renderChat();
}
function renderChat() {
  const body = $("#chat-body");
  body.innerHTML = State.chat.map((m) =>
    `<div class="bubble ${m.role === "user" ? "me" : "bot"}">${esc(m.content)}</div>`).join("");
  body.scrollTop = body.scrollHeight;
}
async function onChatSubmit(e) {
  e.preventDefault();
  const input = $("#chat-text");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  State.chat.push({ role: "user", content: text });
  renderChat();
  State.chat.push({ role: "assistant", content: "…" });
  renderChat();
  try {
    const data = await api("POST", "/api/chat", {
      case_id: State.activeRun ? State.activeRun.case.case_id : null,
      messages: State.chat.filter((m) => m.content !== "…"),
    });
    State.chat[State.chat.length - 1] = { role: "assistant", content: data.reply };
  } catch (err) {
    State.chat[State.chat.length - 1] = { role: "assistant", content: "Sorry — I couldn't reach the assistant. " + err.message };
  }
  renderChat();
}

document.addEventListener("DOMContentLoaded", boot);
