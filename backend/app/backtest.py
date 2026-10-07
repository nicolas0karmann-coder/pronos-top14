"""
Backtest « à l'aveugle » : pour chaque journée, on entraîne le modèle
uniquement sur les matchs joués AVANT cette journée, puis on compare ses
prédictions aux résultats réels.

Usage : python -m app.backtest [--grid]
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
from scipy.stats import norm

from .data import load_dataset, season_label
from .model import ModelConfig, ScoreModel

TEST_SEASONS = [2022, 2023, 2024, 2025, 2026]


def outcome(m):
    return 0 if m.home_score > m.away_score else (1 if m.home_score == m.away_score else 2)


def rounds_of(ds, season):
    by_round = defaultdict(list)
    for m in ds.season(season):
        if m.played and not m.knockout:
            by_round[m.round].append(m)
    return [by_round[r] for r in sorted(by_round)]


def backtest_model(ds, cfg: ModelConfig, seasons=TEST_SEASONS):
    preds = []
    for s in seasons:
        teams = ds.teams(s)
        for rnd in rounds_of(ds, s):
            as_of = min(m.date for m in rnd)
            model = ScoreModel(cfg).fit(ds.matches, as_of, s, teams)
            for m in rnd:
                p = model.predict(m.home, m.away, m.neutral, s)
                preds.append((m, p))
    return preds


class MarginElo:
    """Référence simple : Elo en points d'écart."""

    def __init__(self, k=0.06, hfa=9.0, carry=0.7, promo=-6.0, sigma=15.5, cap=40):
        self.k, self.hfa, self.carry, self.promo, self.sigma, self.cap = k, hfa, carry, promo, sigma, cap

    def run(self, ds, seasons=TEST_SEASONS):
        R, preds, cur_season = {}, [], None
        teams_prev = set()
        for m in ds.played():
            if m.season != cur_season:
                teams_prev = set(ds.teams(cur_season)) if cur_season else set()
                cur_season = m.season
                for t in R:
                    R[t] *= self.carry
                for t in ds.teams(m.season):
                    if t not in teams_prev and cur_season - 1 in {x.season for x in ds.matches}:
                        R[t] = self.promo
            rh, ra = R.setdefault(m.home, self.promo), R.setdefault(m.away, self.promo)
            mu = rh - ra + (0 if m.neutral else self.hfa)
            if m.season in seasons and not m.knockout:
                ph = 1 - norm.cdf(0.5, mu, self.sigma)
                pa = norm.cdf(-0.5, mu, self.sigma)
                preds.append((m, {"p_home": ph, "p_draw": 1 - ph - pa, "p_away": pa,
                                  "expected_margin": mu}))
            err = max(-self.cap, min(self.cap, m.home_score - m.away_score)) - mu
            R[m.home] = rh + self.k * err
            R[m.away] = ra - self.k * err
        return preds


def metrics(preds):
    ll = br = acc = mae = 0.0
    for m, p in preds:
        probs = np.clip([p["p_home"], p["p_draw"], p["p_away"]], 1e-6, 1)
        o = outcome(m)
        ll -= math.log(probs[o])
        y = np.zeros(3); y[o] = 1
        br += ((probs - y) ** 2).sum()
        acc += int(np.argmax(probs) == o)
        mae += abs((m.home_score - m.away_score) - p["expected_margin"])
    n = len(preds)
    bias = float(np.mean([(m.home_score - m.away_score) - p["expected_margin"] for m, p in preds]))
    return {"n": n, "logloss": round(ll / n, 4), "brier": round(float(br / n), 4),
            "accuracy": round(acc / n, 4), "margin_mae": round(mae / n, 2), "margin_bias": round(bias, 2)}


def bonus_metrics(preds):
    """Calibration des probabilités de bonus offensif et défensif."""
    ll = 0.0
    obs = exp = 0.0
    obs_d = exp_d = 0.0
    n = 0
    for m, p in preds:
        if not m.has_detail or "bonus" not in p:
            continue
        b = p["bonus"]
        for side in ("home", "away"):
            won = (m.home_score > m.away_score) if side == "home" else (m.away_score > m.home_score)
            tg = (m.home_tries - m.away_tries) * (1 if side == "home" else -1)
            margin = (m.home_score - m.away_score) * (1 if side == "home" else -1)
            y = 1 if (won and tg >= 3) else 0
            yd = 1 if (-5 <= margin < 0) else 0
            q = min(max(b[f"offensif_{side}"], 1e-6), 1 - 1e-6)
            ll -= y * math.log(q) + (1 - y) * math.log(1 - q)
            obs += y; exp += q
            obs_d += yd; exp_d += b[f"defensif_{side}"]
            n += 1
    return {"bonus_off_logloss": ll / n, "bonus_off_obs": obs / n, "bonus_off_pred": exp / n,
            "bonus_def_obs": obs_d / n, "bonus_def_pred": exp_d / n}


def calibration(preds, bins=(0, .2, .4, .6, .8, .9, 1.01)):
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [(m, p) for m, p in preds if lo <= p["p_home"] < hi]
        if sel:
            rows.append((f"{lo:.1f}-{min(hi,1):.1f}", len(sel),
                         round(np.mean([p["p_home"] for _, p in sel]), 3),
                         round(np.mean([m.home_score > m.away_score for m, _ in sel]), 3)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()
    ds = load_dataset()

    base = [(m, None) for s in TEST_SEASONS for r in rounds_of(ds, s) for m in r]
    freq = np.mean([[outcome(m) == k for k in range(3)] for m, _ in base], 0)
    print("Référence « fréquences moyennes » :",
          metrics([(m, {"p_home": freq[0], "p_draw": freq[1], "p_away": freq[2], "expected_margin": 9.0}) for m, _ in base]))
    print("Référence Elo à écart :", metrics(MarginElo().run(ds)))

    configs = [ModelConfig()]
    if args.grid:
        configs = [
            ModelConfig(half_life_days=hl, ridge=r, home_ridge=hr, team_home=th)
            for hl in (240, 365, 540)
            for r in (1.5, 3.0, 6.0)
            for hr, th in ((15.0, True), (40.0, True), (0.0, False))
        ]
    for cfg in configs:
        preds = backtest_model(ds, cfg)
        print(cfg, metrics(preds), bonus_metrics(preds))
    if not args.grid:
        print("Calibration (p victoire dom. prédite vs observée) :")
        for row in calibration(preds):
            print("  ", row)
        for s in TEST_SEASONS:
            sp = [(m, p) for m, p in preds if m.season == s]
            print(season_label(s), metrics(sp))


if __name__ == "__main__":
    main()
