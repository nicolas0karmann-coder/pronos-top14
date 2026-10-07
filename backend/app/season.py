"""
Classement officiel calculé à partir des résultats, et simulation Monte
Carlo de la fin de saison régulière.

Barème Top 14 : victoire 4, nul 2, défaite 0 ; +1 bonus offensif pour une
victoire avec au moins 3 essais de plus que l'adversaire ; +1 bonus
défensif pour une défaite de 5 points ou moins.
Format : 1er et 2e qualifiés directement en demi-finales, 3e à 6e en
barrages ; 13e en match d'accession contre un club de Pro D2 ; 14e relégué.
Départage simplifié : points, puis différence de points, puis essais
marqués (le règlement officiel commence par les confrontations directes).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .data import Match
from .model import DEF_BONUS_MARGIN, DRAW_PTS, OFF_BONUS_TRY_GAP, WIN_PTS, ScoreModel


def league_points(hs, as_, ht, at):
    """Points de classement (dom, ext). ht/at = essais (None si inconnus)."""
    if hs > as_:
        ph, pa = WIN_PTS, (1 if hs - as_ <= DEF_BONUS_MARGIN else 0)
    elif hs < as_:
        ph, pa = (1 if as_ - hs <= DEF_BONUS_MARGIN else 0), WIN_PTS
    else:
        ph = pa = DRAW_PTS
    bo_h = bo_a = 0
    if ht is not None and at is not None:
        if hs > as_ and ht - at >= OFF_BONUS_TRY_GAP:
            bo_h = 1
        if as_ > hs and at - ht >= OFF_BONUS_TRY_GAP:
            bo_a = 1
    return ph + bo_h, pa + bo_a, bo_h, bo_a


def standings(matches: list[Match], teams: list[str]) -> list[dict]:
    t = {x: defaultdict(int) for x in teams}
    form = defaultdict(list)
    for m in sorted(matches, key=lambda x: x.date):
        if not m.played or m.knockout:
            continue
        ph, pa, bo_h, bo_a = league_points(m.home_score, m.away_score, m.home_tries_any, m.away_tries_any)
        for team, gf, ga, tf, pts, bo, opp_pts in (
            (m.home, m.home_score, m.away_score, m.home_tries_any, ph, bo_h, pa),
            (m.away, m.away_score, m.home_score, m.away_tries_any, pa, bo_a, ph),
        ):
            r = t.setdefault(team, defaultdict(int))
            r["played"] += 1
            r["for"] += gf
            r["against"] += ga
            r["tries"] += tf or 0
            r["points"] += pts
            r["bonus_off"] += bo
            if gf > ga:
                r["won"] += 1
                form[team].append("V")
            elif gf < ga:
                r["lost"] += 1
                if ga - gf <= DEF_BONUS_MARGIN:
                    r["bonus_def"] += 1
                form[team].append("D")
            else:
                r["drawn"] += 1
                form[team].append("N")
    rows = []
    for team, r in t.items():
        rows.append({
            "team": team,
            "played": r["played"], "won": r["won"], "drawn": r["drawn"], "lost": r["lost"],
            "for": r["for"], "against": r["against"], "diff": r["for"] - r["against"],
            "tries": r["tries"], "bonus_off": r["bonus_off"], "bonus_def": r["bonus_def"],
            "points": r["points"], "form": form[team][-5:],
        })
    rows.sort(key=lambda x: (-x["points"], -x["diff"], -x["tries"], x["team"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def simulate_season(model: ScoreModel, season_matches: list[Match], teams: list[str],
                    n_sims: int = 10000, seed: int = 42) -> dict:
    """Simule les matchs de saison régulière restants n_sims fois."""
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(teams)}
    nT = len(teams)
    base_pts = np.zeros(nT)
    base_diff = np.zeros(nT)
    base_tries = np.zeros(nT)
    current = standings(season_matches, teams)
    for r in current:
        i = idx[r["team"]]
        base_pts[i], base_diff[i], base_tries[i] = r["points"], r["diff"], r["tries"]

    remaining = [m for m in season_matches if not m.played and not m.knockout]
    pts = np.tile(base_pts, (n_sims, 1))
    diff = np.tile(base_diff, (n_sims, 1)).astype(float)
    tries = np.tile(base_tries, (n_sims, 1)).astype(float)

    if remaining:
        rates = [model.rates(m.home, m.away, m.neutral) for m in remaining]
        lth = [r["tries"][0] for r in rates]
        lta = [r["tries"][1] for r in rates]
        lkh = [r["kicks"][0] for r in rates]
        lka = [r["kicks"][1] for r in rates]
        s = model.sample(lth, lta, lkh, lka, n_sims, rng)
        hp, ap, ht, at = s["home_pts"], s["away_pts"], s["home_tries"], s["away_tries"]
        home_win, away_win, draw = hp > ap, ap > hp, hp == ap
        margin = hp - ap
        php = (WIN_PTS * home_win + DRAW_PTS * draw
               + (away_win & (-margin <= DEF_BONUS_MARGIN))
               + (home_win & (ht - at >= OFF_BONUS_TRY_GAP)))
        pap = (WIN_PTS * away_win + DRAW_PTS * draw
               + (home_win & (margin <= DEF_BONUS_MARGIN))
               + (away_win & (at - ht >= OFF_BONUS_TRY_GAP)))
        for k, m in enumerate(remaining):
            h, a = idx[m.home], idx[m.away]
            pts[:, h] += php[k]
            pts[:, a] += pap[k]
            diff[:, h] += margin[k]
            diff[:, a] -= margin[k]
            tries[:, h] += ht[k]
            tries[:, a] += at[k]

    # classement de chaque simulation (points, puis différence, puis essais)
    key = pts * 1e8 + (diff + 5000) * 1e3 + tries + rng.random(pts.shape) * 0.1
    order = np.argsort(-key, axis=1)
    ranks = np.empty_like(order)
    rows_idx = np.arange(n_sims)[:, None]
    ranks[rows_idx, order] = np.arange(nT)[None, :]

    out = []
    for t in teams:
        i = idx[t]
        r = ranks[:, i]
        hist = np.bincount(r, minlength=nT) / n_sims
        out.append({
            "team": t,
            "current_points": int(base_pts[i]),
            "expected_points": round(float(pts[:, i].mean()), 1),
            "points_p10": int(np.percentile(pts[:, i], 10)),
            "points_p90": int(np.percentile(pts[:, i], 90)),
            "expected_rank": round(float(r.mean() + 1), 1),
            "p_first": float(hist[0]),
            "p_top2": float(hist[:2].sum()),
            "p_top6": float(hist[:6].sum()),
            "p_13th": float(hist[12]) if nT >= 13 else 0.0,
            "p_14th": float(hist[13]) if nT >= 14 else 0.0,
            "rank_distribution": [round(float(x), 4) for x in hist],
        })
    out.sort(key=lambda x: (-x["expected_points"], x["team"]))
    return {"n_sims": n_sims, "remaining_matches": len(remaining), "teams": out}
