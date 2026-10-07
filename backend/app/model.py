"""
Modèle de prédiction des matchs de Top 14.

Pourquoi pas Dixon-Coles comme pour le foot ?
Au rugby on ne peut pas mettre une loi de Poisson directement sur le score
(19, 27, 40 points...) : les points arrivent par paquets de 3, 5 et 7. En
revanche, les *essais* et les *coups de pied réussis* (pénalités + drops) sont
des comptages qui se comportent très bien comme des lois de Poisson. On a donc
deux sous-modèles « à la Dixon-Coles », un pour les essais et un pour les
coups de pied :

    log(λ_essais dom.) = μ + avantage_dom + bonus_dom[équipe] + attaque[dom] - défense[ext]
    log(λ_essais ext.) = μ + attaque[ext] - défense[dom]

(idem pour les coups de pied), et chaque essai est transformé avec une
probabilité p (≈ 0,77, estimée sur les données).

On reconstitue ensuite exactement la loi du score de chaque équipe
(5 × essais + 2 × transformations + 3 × coups de pied), d'où :
victoire / nul / défaite, score moyen, et surtout les bonus du Top 14,
qui dépendent de l'écart d'essais (offensif) et de l'écart de points
(défensif).

Les matchs anciens comptent moins (demi-vie réglable), et les forces de
chaque équipe sont « rétrécies » vers la moyenne (pénalité ridge), ce qui
évite des valeurs extrêmes pour les équipes avec peu de matchs (promus).
Un avantage du terrain propre à chaque équipe est aussi estimé, avec un
rétrécissement plus fort : en Top 14 certains clubs sont quasi imbattables
chez eux.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom, poisson

from .data import Match

T_MAX = 14  # essais max considérés par équipe
K_MAX = 14  # coups de pied max
PTS_MAX = 5 * T_MAX + 2 * T_MAX + 3 * K_MAX

# règles actuelles du Top 14
WIN_PTS, DRAW_PTS = 4, 2
OFF_BONUS_TRY_GAP = 3  # victoire avec au moins 3 essais de plus
DEF_BONUS_MARGIN = 5   # défaite de 5 points ou moins


@dataclass
class ModelConfig:
    half_life_days: float = 270.0
    ridge: float = 20.0         # rétrécissement attaque/défense
    home_ridge: float = 40.0    # rétrécissement de l'avantage du terrain par équipe
    team_home: bool = True
    promoted_prior: bool = True  # les promus démarrent sous la moyenne
    # demi-vie (jours) pour le niveau général du championnat (μ et avantage
    # du terrain), plus courte : les scores et l'avantage du terrain
    # évoluent d'une saison à l'autre. None = même pondération que le reste.
    global_half_life_days: float | None = 90.0


class ScoreModel:
    def __init__(self, config: ModelConfig | None = None):
        self.cfg = config or ModelConfig()
        self.teams: list[str] = []
        self.idx: dict[str, int] = {}
        self.params: dict[str, dict] = {}
        self.conv_rate = 0.765
        self.as_of: datetime | None = None
        self.n_matches = 0
        self.season: int | None = None
        self.season_teams: dict[int, set] = {}
        self.beta = 0.0   # dépendance coups de pied | essais
        self.kappa = 0.0  # corrélation des essais entre les deux équipes
        self.spread = 1.0  # resserrement de la loi de l'écart (calibré)

    # ------------------------------------------------------------------ fit
    def fit(self, matches: list[Match], as_of: datetime, season: int, season_teams: list[str] | None = None):
        """season : saison (année de début) pour laquelle on va prédire."""
        cfg = self.cfg
        self.season = season
        data = [m for m in matches if m.played and m.has_detail and m.date < as_of]
        if len(data) < 50:
            raise RuntimeError("Pas assez de matchs pour ajuster le modèle.")
        self.as_of = as_of
        self.n_matches = len(data)
        teams = sorted({m.home for m in data} | {m.away for m in data} | set(season_teams or []))
        self.teams = teams
        self.idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = np.array([self.idx[m.home] for m in data])
        ai = np.array([self.idx[m.away] for m in data])
        home_flag = np.array([0.0 if m.neutral else 1.0 for m in data])
        age = np.array([(as_of - m.date).total_seconds() / 86400 for m in data])
        w = np.exp(-np.log(2) * age / cfg.half_life_days)

        tries_conv = sum(m.home_conv + m.away_conv for m in data)
        tries_tot = sum(m.home_tries + m.away_tries for m in data)
        self.conv_rate = tries_conv / max(tries_tot, 1)

        # « promu » = équipe absente de la saison précédente. L'effet (en
        # attaque et en défense) est appris sur tous les promus de l'historique.
        self.season_teams = {}
        for m in matches:
            self.season_teams.setdefault(m.season, set()).update((m.home, m.away))
        ph = np.array([self.is_promoted(m.home, m.season) for m in data], float)
        pa_ = np.array([self.is_promoted(m.away, m.season) for m in data], float)
        if not cfg.promoted_prior:
            ph[:] = 0
            pa_[:] = 0
        self.priors = {}

        self.params = {}
        for comp, yh, ya in (
            ("tries", [m.home_tries for m in data], [m.away_tries for m in data]),
            ("kicks", [m.home_kicks for m in data], [m.away_kicks for m in data]),
        ):
            self.params[comp] = self._fit_component(
                np.array(yh, float), np.array(ya, float), hi, ai, home_flag, w, n, ph, pa_
            )
        if cfg.global_half_life_days:
            wg = np.exp(-np.log(2) * age / cfg.global_half_life_days)
            for comp, yh, ya in (
                ("tries", [m.home_tries for m in data], [m.away_tries for m in data]),
                ("kicks", [m.home_kicks for m in data], [m.away_kicks for m in data]),
            ):
                self._refit_global(comp, np.array(yh, float), np.array(ya, float), hi, ai, home_flag, wg, ph, pa_)
        self._fit_dependence(data, hi, ai, home_flag, w, ph, pa_)
        return self

    def _refit_global(self, comp, yh, ya, hi, ai, hf, w, ph, pa_):
        """Réajuste μ et l'avantage du terrain avec une pondération plus
        récente, à forces d'équipes fixées (quelques itérations de Newton)."""
        p = self.params[comp]
        att, dfn, ht = p["att"], p["def"], p["home_team"]
        oh = att[hi] + p["promo_att"] * ph - dfn[ai] - p["promo_def"] * pa_ + hf * ht[hi]
        oa = att[ai] + p["promo_att"] * pa_ - dfn[hi] - p["promo_def"] * ph
        mu, home = p["mu"], p["home"]
        # léger rappel vers les valeurs long terme (équivalent ~30 matchs)
        mu0, home0, prior_w = mu, home, 30.0 * w.max()
        for _ in range(25):
            lh = np.exp(mu + hf * home + oh)
            la = np.exp(mu + oa)
            g_mu = (w * (lh - yh)).sum() + (w * (la - ya)).sum() + prior_w * (mu - mu0)
            g_h = (w * (lh - yh) * hf).sum() + prior_w * (home - home0)
            h_mm = (w * lh).sum() + (w * la).sum() + prior_w
            h_hh = (w * lh * hf).sum() + prior_w
            h_mh = (w * lh * hf).sum()
            det = h_mm * h_hh - h_mh * h_mh
            mu -= (h_hh * g_mu - h_mh * g_h) / det
            home -= (-h_mh * g_mu + h_mm * g_h) / det
        p["mu"], p["home"] = float(mu), float(home)

    def _train_rates(self, comp, hi, ai, hf, ph, pa_):
        p = self.params[comp]
        att, dfn, ht = p["att"], p["def"], p["home_team"]
        lh = np.exp(p["mu"] + hf * (p["home"] + ht[hi]) + att[hi] + p["promo_att"] * ph - dfn[ai] - p["promo_def"] * pa_)
        la = np.exp(p["mu"] + att[ai] + p["promo_att"] * pa_ - dfn[hi] - p["promo_def"] * ph)
        return lh, la

    def _fit_dependence(self, data, hi, ai, hf, w, ph, pa_):
        """Deux dépendances observées dans les données, ignorées par des
        Poisson indépendantes :
        - beta  : plus une équipe marque d'essais, moins elle passe de
          pénalités (elle joue les touches, et le temps de jeu est limité) ;
        - kappa : les essais des deux équipes sont positivement corrélés
          (matchs « ouverts » vs matchs fermés) -> Poisson bivariée."""
        lth, lta = self._train_rates("tries", hi, ai, hf, ph, pa_)
        lkh, lka = self._train_rates("kicks", hi, ai, hf, ph, pa_)
        Th = np.array([m.home_tries for m in data], float)
        Ta = np.array([m.away_tries for m in data], float)
        Kh = np.array([m.home_kicks for m in data], float)
        Ka = np.array([m.away_kicks for m in data], float)
        T = np.concatenate([Th, Ta]); K = np.concatenate([Kh, Ka])
        LT = np.concatenate([lth, lta]); LK = np.concatenate([lkh, lka]); W = np.concatenate([w, w])

        def nll_beta(b):
            mu = LK * np.exp(b * T - LT * (np.exp(b) - 1))
            return float(np.sum(W * (mu - K * np.log(mu))))

        from scipy.optimize import minimize_scalar
        self.beta = float(minimize_scalar(nll_beta, bounds=(-0.6, 0.3), method="bounded",
                                          options={"xatol": 1e-3}).x)

        from scipy.special import gammaln

        def nll_kappa(k):
            l3 = k * np.minimum(lth, lta)
            l1, l2 = lth - l3, lta - l3
            tot = np.zeros_like(lth)
            for j in range(int(min(Th.max(), Ta.max())) + 1):
                ok = (Th >= j) & (Ta >= j)
                a, b = np.maximum(Th - j, 0), np.maximum(Ta - j, 0)
                term = np.exp(-l1 - l2 - l3 + a * np.log(l1) - gammaln(a + 1)
                              + b * np.log(l2) - gammaln(b + 1) + j * np.log(np.maximum(l3, 1e-12)) - gammaln(j + 1))
                tot += np.where(ok, term, 0.0)
            return float(-np.sum(w * np.log(np.maximum(tot, 1e-300))))

        self.kappa = float(minimize_scalar(nll_kappa, bounds=(0.0, 0.7), method="bounded",
                                           options={"xatol": 2e-3}).x)

        # Dispersion résiduelle : d'autres dépendances (un match « ouvert »
        # augmente les essais ET réduit les pénalités des deux équipes)
        # resserrent encore les écarts réels. On mesure le rapport entre
        # l'écart-type réel des écarts (autour de la prédiction) et celui du
        # modèle, sur les matchs d'entraînement pondérés, et on resserre la
        # loi de l'écart d'autant.
        self.spread = 1.0
        rng = np.random.default_rng(0)
        sims = self.sample(lth, lta, lkh, lka, 400, rng, raw=True)
        m_h, m_a = sims["home_pts"].mean(1), sims["away_pts"].mean(1)
        model_var = (sims["home_pts"] - sims["away_pts"]).var(1)
        obs = np.array([x.home_score - x.away_score for x in data], float)
        resid = obs - (m_h - m_a)
        self.spread = float(np.clip(np.sqrt((w * resid ** 2).sum() / (w * model_var).sum()), 0.6, 1.2))

    # --------------------------------------------------------- simulation
    def sample(self, lth, lta, lkh, lka, n, rng, raw=False):
        """Tire n scores pour chaque match (tableaux de λ de même taille).
        Renvoie essais et points de chaque équipe, de forme (matchs, n)."""
        lth, lta, lkh, lka = (np.asarray(x, float)[:, None] for x in (lth, lta, lkh, lka))
        shape = (lth.shape[0], n)
        l3 = self.kappa * np.minimum(lth, lta)
        x3 = rng.poisson(np.broadcast_to(l3, shape))
        th = rng.poisson(np.broadcast_to(lth - l3, shape)) + x3
        ta = rng.poisson(np.broadcast_to(lta - l3, shape)) + x3
        ch = rng.binomial(th, self.conv_rate)
        ca = rng.binomial(ta, self.conv_rate)
        eb = np.exp(self.beta) - 1
        kh = rng.poisson(lkh * np.exp(self.beta * th - lth * eb))
        ka = rng.poisson(lka * np.exp(self.beta * ta - lta * eb))
        hp = 5 * th + 2 * ch + 3 * kh
        ap = 5 * ta + 2 * ca + 3 * ka
        if not raw and self.spread != 1.0:
            # resserrement autour de la moyenne de chaque équipe
            mh = hp.mean(1, keepdims=True)
            ma = ap.mean(1, keepdims=True)
            hp = np.maximum(np.rint(mh + self.spread * (hp - mh)), 0).astype(int)
            ap = np.maximum(np.rint(ma + self.spread * (ap - ma)), 0).astype(int)
        return {"home_tries": th, "away_tries": ta, "home_pts": hp, "away_pts": ap}

    def is_promoted(self, team: str, season: int) -> float:
        prev = self.season_teams.get(season - 1)
        if not prev:
            return 0.0  # saison précédente absente des données : inconnu
        return 0.0 if team in prev else 1.0

    def _fit_component(self, yh, ya, hi, ai, hf, w, n, ph, pa_):
        cfg = self.cfg
        # vecteur : mu, home, promo_att, promo_def, att[n], def[n], hteam[n]
        n_glob = 4

        def unpack(x):
            mu, home, pa, pd = x[:4]
            att = x[n_glob:n_glob + n]
            dfn = x[n_glob + n:n_glob + 2 * n]
            ht = x[n_glob + 2 * n:n_glob + 3 * n] if cfg.team_home else np.zeros(n)
            return mu, home, pa, pd, att, dfn, ht

        def f(x):
            mu, home, pa, pd, att, dfn, ht = unpack(x)
            eta_h = mu + hf * (home + ht[hi]) + att[hi] + pa * ph - dfn[ai] - pd * pa_
            eta_a = mu + att[ai] + pa * pa_ - dfn[hi] - pd * ph
            lh, la = np.exp(eta_h), np.exp(eta_a)
            nll = np.sum(w * (lh - yh * eta_h)) + np.sum(w * (la - ya * eta_a))
            pen = cfg.ridge * (att @ att + dfn @ dfn)
            if cfg.team_home:
                pen += cfg.home_ridge * (ht @ ht)
            pen += 0.1 * (pa * pa + pd * pd)
            # gradients
            rh, ra = w * (lh - yh), w * (la - ya)
            g = np.zeros_like(x)
            g[0] = rh.sum() + ra.sum()
            g[1] = (rh * hf).sum()
            g_att = np.bincount(hi, rh, n) + np.bincount(ai, ra, n)
            g_def = -np.bincount(ai, rh, n) - np.bincount(hi, ra, n)
            g[2] = (rh * ph).sum() + (ra * pa_).sum() + 0.2 * pa
            g[3] = -(rh * pa_).sum() - (ra * ph).sum() + 0.2 * pd
            g[n_glob:n_glob + n] = g_att + 2 * cfg.ridge * att
            g[n_glob + n:n_glob + 2 * n] = g_def + 2 * cfg.ridge * dfn
            if cfg.team_home:
                g[n_glob + 2 * n:] = np.bincount(hi, rh * hf, n) + 2 * cfg.home_ridge * ht
            return nll + pen, g

        size = n_glob + (3 if cfg.team_home else 2) * n
        x0 = np.zeros(size)
        x0[0] = np.log(max((yh.sum() + ya.sum()) / (2 * len(yh)), 0.1))
        res = minimize(f, x0, jac=True, method="L-BFGS-B")
        mu, home, pa, pd, att, dfn, ht = unpack(res.x)
        return {
            "mu": float(mu), "home": float(home), "promo_att": float(pa), "promo_def": float(pd),
            "att": att, "def": dfn, "home_team": ht,
        }

    # ------------------------------------------------------------- predict
    def _team_param(self, comp, key, team, season=None):
        p = self.params[comp]
        v = float(p[key][self.idx[team]]) if team in self.idx else 0.0
        if key in ("att", "def") and season is not None:
            promo = self.is_promoted(team, season) if team in self.idx else 1.0
            v += promo * p["promo_att" if key == "att" else "promo_def"]
        return v

    def rates(self, home: str, away: str, neutral: bool = False, season: int | None = None):
        """λ essais et λ coups de pied pour chaque équipe."""
        season = self.season if season is None else season
        out = {}
        for comp in ("tries", "kicks"):
            p = self.params[comp]
            h_adv = 0.0 if neutral else p["home"] + self._team_param(comp, "home_team", home)
            out[comp] = (
                float(np.exp(p["mu"] + h_adv + self._team_param(comp, "att", home, season) - self._team_param(comp, "def", away, season))),
                float(np.exp(p["mu"] + self._team_param(comp, "att", away, season) - self._team_param(comp, "def", home, season))),
            )
        return out

    def conditional_points(self, lam_t: float, lam_k: float) -> np.ndarray:
        """Q[T, pts] = P(points = pts | essais = T) pour une équipe."""
        Q = np.zeros((T_MAX + 1, PTS_MAX + 1))
        p = self.conv_rate
        for t in range(T_MAX + 1):
            lk = lam_k * np.exp(self.beta * t - lam_t * (np.exp(self.beta) - 1))
            pk = poisson.pmf(np.arange(K_MAX + 1), lk)
            pc = binom.pmf(np.arange(t + 1), t, p)
            for c in range(t + 1):
                base = 5 * t + 2 * c
                Q[t, base:base + 3 * (K_MAX + 1):3] += pc[c] * pk
            Q[t] /= Q[t].sum()
        return Q

    def joint_tries(self, lh: float, la: float) -> np.ndarray:
        """J[Th, Ta] : loi de Poisson bivariée des essais."""
        l3 = self.kappa * min(lh, la)
        r = np.arange(T_MAX + 1)
        p1, p2, p3 = poisson.pmf(r, lh - l3), poisson.pmf(r, la - l3), poisson.pmf(r, l3)
        J = np.zeros((T_MAX + 1, T_MAX + 1))
        for k in range(T_MAX + 1):
            J[k:, k:] += p3[k] * np.outer(p1[: T_MAX + 1 - k], p2[: T_MAX + 1 - k])
        return J / J.sum()

    def _compress(self, pmf, d):
        """Resserre une loi discrète autour de sa moyenne d'un facteur spread."""
        if self.spread == 1.0:
            return pmf
        mu = float((d * pmf).sum())
        edges = np.concatenate([[d[0] - 0.5], d + 0.5])
        cdf = np.concatenate([[0.0], np.cumsum(pmf)])
        src = mu + (edges - mu) / self.spread
        new_cdf = np.interp(src, edges, cdf, left=0.0, right=1.0)
        out = np.diff(new_cdf)
        return out / out.sum()

    def predict(self, home: str, away: str, neutral: bool = False, season: int | None = None) -> dict:
        r = self.rates(home, away, neutral, season)
        (lth, lta), (lkh, lka) = r["tries"], r["kicks"]
        QH = self.conditional_points(lth, lkh)
        QA = self.conditional_points(lta, lka)
        J = self.joint_tries(lth, lta)

        # loi de l'écart de points (dom - ext) : mélange sur les essais
        d = np.arange(-PTS_MAX, PTS_MAX + 1)
        diff = np.zeros(2 * PTS_MAX + 1)
        QA_rev = QA[:, ::-1]
        for th in range(T_MAX + 1):
            if J[th].sum() < 1e-12:
                continue
            for ta in range(T_MAX + 1):
                if J[th, ta] < 1e-12:
                    continue
                diff += J[th, ta] * np.convolve(QH[th], QA_rev[ta])
        diff /= diff.sum()
        diff = self._compress(diff, d)
        p_home = float(diff[d > 0].sum())
        p_draw = float(diff[d == 0].sum())
        p_away = float(diff[d < 0].sum())

        # bonus offensif : victoire ET au moins 3 essais de plus
        cdf_lt_A = np.concatenate([np.zeros((T_MAX + 1, 1)), np.cumsum(QA, 1)[:, :-1]], 1)
        cdf_lt_H = np.concatenate([np.zeros((T_MAX + 1, 1)), np.cumsum(QH, 1)[:, :-1]], 1)
        win_h = (QH @ cdf_lt_A.T) * J      # [Th, Ta]
        win_a = (QA @ cdf_lt_H.T) * J.T    # [Ta, Th]
        tg = np.subtract.outer(np.arange(T_MAX + 1), np.arange(T_MAX + 1))
        bo_home = float(win_h[tg >= OFF_BONUS_TRY_GAP].sum())
        bo_away = float(win_a[tg >= OFF_BONUS_TRY_GAP].sum())
        bd_home = float(diff[(d < 0) & (d >= -DEF_BONUS_MARGIN)].sum())
        bd_away = float(diff[(d > 0) & (d <= DEF_BONUS_MARGIN)].sum())

        pts = np.arange(PTS_MAX + 1)
        ph = J.sum(1) @ QH
        pa = J.sum(0) @ QA
        mean_h = float((np.arange(PTS_MAX + 1) * ph).sum())
        mean_a = float((np.arange(PTS_MAX + 1) * pa).sum())
        margin_sd = float(np.sqrt((d * d * diff).sum() - (d * diff).sum() ** 2))

        def cumul(lo, hi):
            return float(diff[(d >= lo) & (d <= hi)].sum())

        return {
            "home": home,
            "away": away,
            "neutral": neutral,
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "expected_score": [round(mean_h, 1), round(mean_a, 1)],
            "expected_tries": [round(lth, 2), round(lta, 2)],
            "expected_margin": round(mean_h - mean_a, 1),
            "margin_sd": round(margin_sd, 1),
            "bonus": {
                "offensif_home": bo_home,
                "offensif_away": bo_away,
                "defensif_home": bd_home,
                "defensif_away": bd_away,
            },
            "expected_league_points": [
                round(WIN_PTS * p_home + DRAW_PTS * p_draw + bo_home + bd_home, 2),
                round(WIN_PTS * p_away + DRAW_PTS * p_draw + bo_away + bd_away, 2),
            ],
            "margin_buckets": {
                "home_by_15_plus": cumul(15, PTS_MAX),
                "home_by_8_14": cumul(8, 14),
                "home_by_1_7": cumul(1, 7),
                "draw": p_draw,
                "away_by_1_7": cumul(-7, -1),
                "away_by_8_14": cumul(-14, -8),
                "away_by_15_plus": cumul(-PTS_MAX, -15),
            },
        }

    # ------------------------------------------------------------- ratings
    def ratings(self, teams: list[str]) -> list[dict]:
        """Indice de force : écart de points attendu contre une équipe moyenne,
        sur terrain neutre (positif = au-dessus de la moyenne)."""
        avg_t = np.exp(self.params["tries"]["mu"])
        avg_k = np.exp(self.params["kicks"]["mu"])
        pts_per_try = 5 + 2 * self.conv_rate
        out = []
        for t in teams:
            r = {}
            for comp, unit, avg in (("tries", pts_per_try, avg_t), ("kicks", 3, avg_k)):
                a = self._team_param(comp, "att", t, self.season)
                df = self._team_param(comp, "def", t, self.season)
                r[comp] = (unit * avg * (np.exp(a) - 1), unit * avg * (np.exp(df) - 1))
            attack = r["tries"][0] + r["kicks"][0]
            defense = r["tries"][1] + r["kicks"][1]
            home_bonus = 0.0
            if t in self.idx:
                for comp, unit, avg in (("tries", pts_per_try, avg_t), ("kicks", 3, avg_k)):
                    hb = self._team_param(comp, "home_team", t)
                    home_bonus += unit * avg * np.exp(self.params[comp]["home"]) * (np.exp(hb) - 1)
            out.append({
                "team": t,
                "attack": round(float(attack), 1),     # points marqués en plus de la moyenne
                "defense": round(float(defense), 1),   # points encaissés en moins de la moyenne
                "rating": round(float(attack + defense), 1),
                "home_extra": round(float(home_bonus), 1),
                "promoted": bool(self.is_promoted(t, self.season)) if self.season else False,
            })
        out.sort(key=lambda x: -x["rating"])
        return out

    def global_home_advantage_points(self) -> float:
        avg_t = np.exp(self.params["tries"]["mu"])
        avg_k = np.exp(self.params["kicks"]["mu"])
        ppt = 5 + 2 * self.conv_rate
        h = ppt * avg_t * (np.exp(self.params["tries"]["home"]) - 1) + 3 * avg_k * (np.exp(self.params["kicks"]["home"]) - 1)
        return round(float(h), 1)
