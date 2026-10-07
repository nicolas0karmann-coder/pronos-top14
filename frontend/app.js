"use strict";

const $ = (s) => document.querySelector(s);
const pct = (x) => `${Math.round(x * 100)} %`;
const pctFine = (x) => (x < 0.01 && x > 0 ? "< 1 %" : pct(x));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Sur l'offre gratuite de Render, le serveur s'endort après 15 min sans visite :
// au réveil il retélécharge les données, ce qui peut prendre un moment.
let slowTimer = null, pending = 0;
function slowHint(on) {
  const el = document.getElementById("slowHint");
  pending += on ? 1 : -1;
  if (on && pending === 1) slowTimer = setTimeout(() => { el.hidden = false; }, 3000);
  if (pending === 0) { clearTimeout(slowTimer); el.hidden = true; }
}

async function api(path, opts) {
  slowHint(true);
  let r;
  try {
    r = await fetch(path, opts);
  } catch (_) {
    slowHint(false);
    throw new Error("Serveur injoignable. Vérifiez votre connexion puis rechargez la page.");
  }
  slowHint(false);
  if (!r.ok) {
    let msg = `Erreur ${r.status}`;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

const dateFmt = new Intl.DateTimeFormat("fr-FR", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "Europe/Paris" });
const dayFmt = new Intl.DateTimeFormat("fr-FR", { day: "numeric", month: "short", year: "numeric", timeZone: "Europe/Paris" });

/* ---------- Barre-terrain ---------- */
// Lignes d'un terrain de 100 m (en-but compris) projetées sur la barre.
const PITCH_LINES = [
  { m: 0, cls: "try" }, { m: 22, cls: "" }, { m: 40, cls: "dash" },
  { m: 50, cls: "" }, { m: 60, cls: "dash" }, { m: 78, cls: "" }, { m: 100, cls: "try" },
];
function pitchBar(p) {
  const seg = (cls, v, label) =>
    `<div class="seg ${cls}" style="flex-basis:${(v * 100).toFixed(2)}%">${v >= 0.12 ? `<span class="lbl">${label}</span>` : ""}</div>`;
  const lines = PITCH_LINES.map((l) => `<span class="line ${l.cls}" style="left:${6 + l.m * 0.88}%"></span>`).join("");
  return `<div class="pitch" role="img" aria-label="Victoire domicile ${pct(p.p_home)}, nul ${pctFine(p.p_draw)}, victoire extérieur ${pct(p.p_away)}">
      ${seg("h", p.p_home, pct(p.p_home))}${seg("d", p.p_draw, "")}${seg("a", p.p_away, pct(p.p_away))}${lines}
    </div>
    <div class="probs-legend"><span>Victoire domicile</span><span>Nul ${pctFine(p.p_draw)}</span><span>Victoire extérieur</span></div>`;
}

function details(p, home, away) {
  const b = p.bonus;
  const row = (label, h, a) => `<tr><td class="hv">${h}</td><th scope="row">${label}</th><td class="av">${a}</td></tr>`;
  const sign = p.expected_margin > 0 ? "+" : "";
  return `<table class="duel" aria-label="Détails ${esc(home)} contre ${esc(away)}">
    ${row("Essais attendus", p.expected_tries[0].toFixed(1), p.expected_tries[1].toFixed(1))}
    ${row("Bonus offensif", pctFine(b.offensif_home), pctFine(b.offensif_away))}
    ${row("Bonus défensif", pctFine(b.defensif_home), pctFine(b.defensif_away))}
    ${row("Points de classement attendus", p.expected_league_points[0].toFixed(1), p.expected_league_points[1].toFixed(1))}
  </table>
  <p class="margin-note">Écart moyen prévu : ${Math.abs(p.expected_margin)} points en faveur de ${esc(p.expected_margin >= 0 ? home : away)}, à ± ${p.margin_sd} points près dans deux matchs sur trois.</p>`;
}

/* ---------- Journée ---------- */
let roundState = { round: null, rounds: [] };

async function loadRound(n) {
  const list = $("#matchList");
  list.innerHTML = `<li class="loading">Calcul des pronostics…</li>`;
  try {
    const j = await api(n == null ? "/api/journee/courante" : `/api/journee/${n}`);
    roundState = { round: j.round, rounds: j.rounds, current: j.current_round };
    $("#roundTitle").textContent = `Journée ${j.round}`;
    $("#prevRound").disabled = j.round <= j.rounds[0];
    $("#nextRound").disabled = j.round >= j.rounds[j.rounds.length - 1];
    let note = j.round === j.current_round ? "Prochaine journée" : (j.round < j.current_round ? "Journée jouée" : "Journée à venir");
    if (j.played) note += `. Pronostic juste sur ${j.correct} match${j.correct > 1 ? "s" : ""} sur ${j.played} , avec un modèle entraîné avant la journée.`;
    $("#roundNote").textContent = note;
    list.innerHTML = j.matches.map(matchCard).join("");
  } catch (e) {
    list.innerHTML = `<li class="error">${esc(e.message)}</li>`;
  }
}

function matchCard(m) {
  const p = m.prediction;
  const played = m.home_score != null;
  const center = played
    ? `<span class="num">${m.home_score} – ${m.away_score}</span><span class="sub">prévu ${Math.round(p.expected_score[0])} – ${Math.round(p.expected_score[1])}</span>`
    : `<span class="num">${Math.round(p.expected_score[0])} – ${Math.round(p.expected_score[1])}</span><span class="sub">score moyen prévu</span>`;
  const verdict = played
    ? `<span class="verdict ${m.prediction_correct ? "ok" : "ko"}">${m.prediction_correct ? "✓ Pronostic juste" : "✗ Pronostic manqué"}</span>`
    : "";
  const tries = played && m.home_tries != null ? `, essais ${m.home_tries} – ${m.away_tries}` : "";
  return `<li class="match">
    <div class="match-head">
      <span class="team home">${esc(m.home)}</span>
      <span class="score">${center}</span>
      <span class="team away">${esc(m.away)}</span>
    </div>
    <div class="meta"><span>${dateFmt.format(new Date(m.date))}${m.stadium ? ", " + esc(m.stadium) : ""}${tries}</span>${verdict}</div>
    ${pitchBar(p)}
    ${details(p, m.home, m.away)}
  </li>`;
}

$("#prevRound").addEventListener("click", () => loadRound(roundState.round - 1));
$("#nextRound").addEventListener("click", () => loadRound(roundState.round + 1));

/* ---------- Classement ---------- */
function zone(rank, n) {
  if (rank <= 2) return "z-semi";
  if (rank <= 6) return "z-barrage";
  if (rank === n - 1) return "z-access";
  if (rank === n) return "z-releg";
  return "";
}

async function loadStandings() {
  const t = $("#standings");
  t.innerHTML = `<tr><td class="loading">Chargement…</td></tr>`;
  try {
    const j = await api("/api/classement");
    const n = j.rows.length;
    t.innerHTML = `<thead><tr>
      <th class="l">#</th><th class="l">Équipe</th><th>J</th><th>V</th><th>N</th><th>D</th>
      <th class="hide-sm">Pour</th><th class="hide-sm">Contre</th><th>Diff</th>
      <th title="Bonus offensifs">BO</th><th title="Bonus défensifs">BD</th><th>Pts</th><th class="l hide-sm">Forme</th>
    </tr></thead><tbody>${j.rows.map((r) => `<tr class="${zone(r.rank, n)}">
      <td class="l rank">${r.rank}</td><td class="l team-cell">${esc(r.team)}</td>
      <td>${r.played}</td><td>${r.won}</td><td>${r.drawn}</td><td>${r.lost}</td>
      <td class="hide-sm">${r.for}</td><td class="hide-sm">${r.against}</td><td>${r.diff > 0 ? "+" : ""}${r.diff}</td>
      <td>${r.bonus_off}</td><td>${r.bonus_def}</td><td class="pts">${r.points}</td>
      <td class="l hide-sm"><span class="form">${r.form.map((f) => `<span class="${f}" title="${{ V: "Victoire", D: "Défaite", N: "Nul" }[f]}">${f}</span>`).join("")}</span></td>
    </tr>`).join("")}</tbody>`;
  } catch (e) {
    t.innerHTML = `<tr><td class="error">${esc(e.message)}</td></tr>`;
  }
}

/* ---------- Projection ---------- */
const bar = (x, color) => `<span class="pbar">${pctFine(x)}<i style="--w:${(x * 100).toFixed(1)}%;${color ? `--c:${color}` : ""}"></i></span>`;

async function loadProjection() {
  const t = $("#projection");
  t.innerHTML = `<tr><td class="loading">Simulation de la saison…</td></tr>`;
  try {
    const j = await api("/api/projection");
    $("#projNote").textContent = `${j.remaining_matches} matchs restants simulés ${j.n_sims.toLocaleString("fr-FR")} fois, bonus compris. La colonne « places » montre la probabilité de finir à chaque rang, du 1er (à gauche) au 14e.`;
    t.innerHTML = `<thead><tr>
      <th class="l">Équipe</th><th>Pts actuels</th><th>Pts prévus</th><th class="hide-sm" title="8 saisons simulées sur 10 finissent dans cet intervalle">Fourchette</th>
      <th>Top 2</th><th>Top 6</th><th>13e</th><th>14e</th><th class="l hide-sm">Places</th>
    </tr></thead><tbody>${j.teams.map((r) => `<tr>
      <td class="l team-cell">${esc(r.team)}</td>
      <td>${r.current_points}</td><td class="pts">${r.expected_points.toFixed(0)}</td>
      <td class="range hide-sm">${r.points_p10}–${r.points_p90}</td>
      <td>${bar(r.p_top2, "var(--z-semi)")}</td><td>${bar(r.p_top6, "var(--z-barrage)")}</td>
      <td>${bar(r.p_13th, "var(--z-access)")}</td><td>${bar(r.p_14th, "var(--z-releg)")}</td>
      <td class="l hide-sm"><span class="heat" aria-label="Rang le plus probable : ${r.rank_distribution.indexOf(Math.max(...r.rank_distribution)) + 1}">${r.rank_distribution.map((x, i) => `<span title="${i + 1}e : ${pctFine(x)}" style="opacity:${Math.max(0.06, Math.min(1, x * 2.5)).toFixed(2)}"></span>`).join("")}</span></td>
    </tr>`).join("")}</tbody>`;
  } catch (e) {
    t.innerHTML = `<tr><td class="error">${esc(e.message)}</td></tr>`;
  }
}

/* ---------- Match au choix ---------- */
const BUCKETS = [
  ["home_by_15_plus", "Dom. +15", "h"], ["home_by_8_14", "Dom. 8-14", "h"], ["home_by_1_7", "Dom. 1-7", "h"],
  ["draw", "Nul", ""], ["away_by_1_7", "Ext. 1-7", "a"], ["away_by_8_14", "Ext. 8-14", "a"], ["away_by_15_plus", "Ext. +15", "a"],
];

async function loadTeams() {
  const j = await api("/api/equipes");
  const opts = j.teams.map((t) => `<option>${esc(t)}</option>`).join("");
  $("#homeSel").innerHTML = opts;
  $("#awaySel").innerHTML = opts;
  $("#homeSel").value = j.teams.includes("Toulouse") ? "Toulouse" : j.teams[0];
  $("#awaySel").value = j.teams.includes("Toulon") ? "Toulon" : j.teams[1];
}

async function runMatch() {
  const home = $("#homeSel").value, away = $("#awaySel").value;
  const out = $("#matchResult");
  if (home === away) { out.innerHTML = `<p class="error">Choisissez deux équipes différentes.</p>`; return; }
  out.innerHTML = `<p class="loading">Calcul…</p>`;
  try {
    const p = await api(`/api/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}&neutre=${$("#neutralChk").checked}`);
    const maxB = Math.max(...BUCKETS.map((b) => p.margin_buckets[b[0]]));
    out.innerHTML = `<div class="match">
      <div class="match-head">
        <span class="team home">${esc(home)}</span>
        <span class="score"><span class="num">${Math.round(p.expected_score[0])} – ${Math.round(p.expected_score[1])}</span><span class="sub">score moyen prévu</span></span>
        <span class="team away">${esc(away)}</span>
      </div>
      ${pitchBar(p)}
      ${details(p, home, away)}
      <h3>Écart final</h3>
      <div class="buckets">${BUCKETS.map(([k, , c]) => `<div class="${c}" style="height:${(p.margin_buckets[k] / maxB * 100).toFixed(1)}%"><span>${pctFine(p.margin_buckets[k])}</span></div>`).join("")}</div>
      <div class="bucket-labels">${BUCKETS.map((b) => `<span>${b[1]}</span>`).join("")}</div>
      <h3>Dernières confrontations</h3>
      ${p.head_to_head.length ? `<ul class="h2h">${p.head_to_head.map((m) => `<li><span>${dayFmt.format(new Date(m.date))}${m.knockout ? " (phase finale)" : ""}</span><span>${esc(m.home)} ${m.home_score} – ${m.away_score} ${esc(m.away)}</span></li>`).join("")}</ul>` : `<p class="muted">Aucune confrontation dans l'historique disponible.</p>`}
    </div>`;
  } catch (e) {
    out.innerHTML = `<p class="error">${esc(e.message)}</p>`;
  }
}
["#homeSel", "#awaySel", "#neutralChk"].forEach((s) => $(s).addEventListener("change", runMatch));
$("#swapBtn").addEventListener("click", () => {
  const h = $("#homeSel").value;
  $("#homeSel").value = $("#awaySel").value;
  $("#awaySel").value = h;
  runMatch();
});

/* ---------- Forces ---------- */
async function loadForces() {
  const box = $("#forces");
  box.innerHTML = `<p class="loading">Chargement…</p>`;
  try {
    const j = await api("/api/forces");
    $("#forcesNote").textContent = `Écart de points attendu contre une équipe moyenne du Top 14, sur terrain neutre. À cela s'ajoute l'avantage du terrain : ${j.home_advantage_points} points en moyenne pour l'équipe qui reçoit, plus ou moins selon le club.`;
    const max = Math.max(...j.rows.map((r) => Math.abs(r.rating)), 1);
    box.innerHTML = j.rows.map((r) => {
      const w = (Math.abs(r.rating) / max) * 50;
      const sign = (x) => (x > 0 ? "+" : "") + x.toFixed(1);
      return `<div class="force-row">
        <span class="name">${esc(r.team)}${r.promoted ? `<span class="tag">promu</span>` : ""}</span>
        <span class="force-track"><span class="bar ${r.rating >= 0 ? "pos" : "neg"}" style="width:${w.toFixed(1)}%"></span></span>
        <span class="val">${sign(r.rating)}</span>
        <span class="force-sub">Attaque ${sign(r.attack)}, défense ${sign(r.defense)}, à domicile ${sign(r.home_extra)} par rapport à la moyenne</span>
      </div>`;
    }).join("");
  } catch (e) {
    box.innerHTML = `<p class="error">${esc(e.message)}</p>`;
  }
}

/* ---------- Onglets ---------- */
const loaded = {};
const loaders = {
  journee: () => loadRound(null),
  classement: loadStandings,
  projection: loadProjection,
  match: async () => { await loadTeams(); runMatch(); },
  forces: loadForces,
};
function show(view) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === view)));
  document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== `view-${view}`));
  if (!loaded[view]) { loaded[view] = true; loaders[view](); }
  history.replaceState(null, "", `#${view}`);
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));

/* ---------- Statut ---------- */
async function loadStatus() {
  try {
    const s = await api("/api/statut");
    $("#seasonLine").textContent = `Saison ${s.season}, résultats à jour au ${dayFmt.format(new Date(s.last_result))}.`;
    $("#footSource").textContent = `Données : ${s.source}. ${s.matches_in_history} matchs d'historique.`;
    const w = $("#warnings");
    w.hidden = !s.warnings.length;
    w.innerHTML = s.warnings.map((x) => `<p>${esc(x)}</p>`).join("");
  } catch (e) {
    $("#seasonLine").textContent = `Données indisponibles : ${e.message}`;
  }
}
$("#refreshBtn").addEventListener("click", async () => {
  $("#refreshBtn").textContent = "Rechargement…";
  try { await api("/api/rafraichir", { method: "POST" }); } catch (_) {}
  Object.keys(loaded).forEach((k) => delete loaded[k]);
  $("#refreshBtn").textContent = "Recharger les données";
  await loadStatus();
  show(location.hash.slice(1) in loaders ? location.hash.slice(1) : "journee");
});

loadStatus();
show(location.hash.slice(1) in loaders ? location.hash.slice(1) : "journee");
