"""
Chargement et nettoyage des données Top 14.

Source : dépôt GitHub transientlunatic/Rugby-Data (licence MIT), mis à jour
automatiquement chaque lundi. Un fichier JSON par saison, avec pour chaque
match : équipes, score, date, stade et, depuis 2014-15, le détail des
actions de score (essais, transformations, pénalités, drops).

Particularités corrigées ici :
- les deux premiers fichiers sont mal nommés ("2011-2010" contient en
  réalité la saison 2011-12, "2012-2011" la saison 2012-13) ;
- avant 2021, l'année des dates est fausse (toujours 2019) : le jour et le
  mois sont justes, on reconstruit l'année à partir de la saison ;
- un même club apparaît sous plusieurs noms selon les saisons
  ("Bordeaux-Beg", "Bordeaux-Begles"...) : on unifie.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

RAW_ROOT = "https://raw.githubusercontent.com/transientlunatic/Rugby-Data/master/json"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_TTL_CURRENT = 6 * 3600  # la saison en cours est rafraîchie toutes les 6 h

# année de début de saison -> nom du fichier dans le dépôt
SEASON_FILES = {
    2011: "top14-2011-2010.json",
    2012: "top14-2012-2011.json",
    2013: "top14-2013-2014.json",
    2014: "top14-2014-2015.json",
    2015: "top14-2015-2016.json",  # incomplète (88 matchs)
    2017: "top14-2017-2018.json",
    2018: "top14-2018-2019.json",
    2021: "top14-2021-2022.json",
    2022: "top14-2022-2023.json",
    2023: "top14-2023-2024.json",
    2024: "top14-2024-2025.json",
    2025: "top14-2025-2026.json",  # s'arrête mi-février 2026 dans la source
    2026: "top14-2026-2027.json",
}

TEAM_NAMES = {
    "Bordeaux-Beg": "Bordeaux-Bègles",
    "Bordeaux-Begles": "Bordeaux-Bègles",
    "Castres Olympique": "Castres",
    "Clermont Auvergne": "Clermont",
    "Lyon O.U.": "Lyon",
    "Racing": "Racing 92",
    "Stade Francais": "Stade Français",
    "Stade Francais Paris": "Stade Français",
    "Biarritz Olympique": "Biarritz",
}


def canonical(name: str) -> str:
    return TEAM_NAMES.get(name, name)


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def current_season_start(today: datetime | None = None) -> int:
    today = today or datetime.now(timezone.utc)
    return today.year if today.month >= 7 else today.year - 1


@dataclass
class Match:
    season: int
    date: datetime
    round: int | None
    home: str
    away: str
    knockout: bool
    neutral: bool
    home_score: int | None = None
    away_score: int | None = None
    # détail des points, None si les événements ne sont pas disponibles
    # ou incohérents avec le score final
    home_tries: int | None = None
    away_tries: int | None = None
    home_kicks: int | None = None  # pénalités + drops réussis
    away_kicks: int | None = None
    home_conv: int | None = None
    away_conv: int | None = None
    # nombre d'essais même quand le détail est incomplet (pour les bonus)
    home_tries_any: int | None = None
    away_tries_any: int | None = None
    # match issu d'un complément (cf. app/supplements) : essais reconstitués,
    # date éventuellement approximative
    reconstructed: bool = False
    date_approx: bool = False
    stadium: str | None = None

    @property
    def played(self) -> bool:
        return self.home_score is not None and self.away_score is not None

    @property
    def has_detail(self) -> bool:
        return self.home_tries is not None and self.away_tries is not None

    def to_dict(self) -> dict:
        return {
            "season": season_label(self.season),
            "date": self.date.isoformat(),
            "round": self.round,
            "home": self.home,
            "away": self.away,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "home_tries": self.home_tries_any,
            "away_tries": self.away_tries_any,
            "knockout": self.knockout,
            "stadium": self.stadium,
            "reconstructed": self.reconstructed,
            "date_approx": self.date_approx,
        }


def _fetch_json(filename: str, ttl: float | None) -> list | None:
    """Télécharge un fichier de saison avec cache disque.
    ttl=None : cache permanent (saison terminée)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / filename
    if path.exists() and (ttl is None or time.time() - path.stat().st_mtime < ttl):
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        r = requests.get(f"{RAW_ROOT}/{filename}", timeout=15)
        r.raise_for_status()
        data = r.json()
        path.write_text(json.dumps(data), encoding="utf-8")
        return data
    except (requests.RequestException, ValueError):
        # réseau indisponible : on se rabat sur le cache même périmé
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None


def _parse_date(raw: str, start_year: int) -> datetime:
    d = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    year = start_year if d.month >= 7 else start_year + 1
    d = d.replace(year=year)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


def _count_tries(side: dict):
    events = side.get("scores") or []
    if not events:
        return None
    n = sum(1 for e in events if "try" in (e.get("type") or "").lower() and (e.get("value") or 0) > 0)
    return n if side.get("score") is not None and 5 * n <= side["score"] else None


def _score_detail(side: dict):
    """(essais, transformations, coups de pied réussis) ou None si le détail
    est absent ou ne retombe pas sur le score final."""
    events = side.get("scores") or []
    if not events or side.get("score") is None:
        return None
    tries = conv = kicks = total = 0
    for e in events:
        t = (e.get("type") or "").lower()
        v = e.get("value") or 0
        total += v
        if v == 0:
            continue
        if "try" in t:
            tries += 1
        elif t == "conversion":
            conv += 1
        elif t in ("penalty", "drop goal"):
            kicks += 1
    if total != side["score"]:
        return None
    return tries, conv, kicks


def parse_season(start_year: int, raw: list) -> list[Match]:
    rows = []
    for m in raw:
        h, a = m["home"], m["away"]
        if h.get("team") == "TBC" or a.get("team") == "TBC":
            continue
        rows.append((m, _parse_date(m["date"], start_year)))
    rows.sort(key=lambda x: x[1])

    has_round_type = any("round_type" in m for m, _ in rows)
    n = len(rows)
    out = []
    games = {}
    for i, (m, d) in enumerate(rows):
        th, ta = canonical(m["home"]["team"]), canonical(m["away"]["team"])
        # Le champ round_type n'est pas toujours renseigné (ou fiable) : un
        # match est aussi considéré comme une phase finale si l'une des deux
        # équipes a déjà joué ses 26 matchs de saison régulière, ou s'il vient
        # après les 182 matchs (26 journées x 7) de la saison régulière.
        ko = (
            m.get("round_type") == "knockout"
            or games.get(th, 0) >= 26
            or games.get(ta, 0) >= 26
            or i >= 182
        )
        if not ko:
            games[th] = games.get(th, 0) + 1
            games[ta] = games.get(ta, 0) + 1
        out.append(
            Match(
                season=start_year,
                date=d,
                round=m.get("round"),
                home=canonical(m["home"]["team"]),
                away=canonical(m["away"]["team"]),
                knockout=ko,
                neutral=False,
                stadium=m.get("stadium"),
            )
        )
        hs, as_ = m["home"].get("score"), m["away"].get("score")
        if hs is not None and as_ is not None:
            out[-1].home_score, out[-1].away_score = int(hs), int(as_)
            out[-1].home_tries_any = _count_tries(m["home"])
            out[-1].away_tries_any = _count_tries(m["away"])
            dh, da = _score_detail(m["home"]), _score_detail(m["away"])
            if dh and da:
                out[-1].home_tries, out[-1].home_conv, out[-1].home_kicks = dh
                out[-1].away_tries, out[-1].away_conv, out[-1].away_kicks = da

    # demi-finales et finale (les 3 derniers matchs de phase finale) se jouent
    # sur terrain neutre ; les barrages chez le mieux classé.
    ko_matches = [x for x in out if x.knockout]
    for x in ko_matches[-3:]:
        x.neutral = True

    # numéro de journée pour les anciennes saisons : on l'infère par blocs de 7
    if not has_round_type:
        league = [x for x in out if not x.knockout]
        for i, x in enumerate(league):
            x.round = i // 7 + 1
    return out


@dataclass
class Dataset:
    matches: list[Match] = field(default_factory=list)
    loaded_at: datetime | None = None
    missing_seasons: list[int] = field(default_factory=list)
    supplemented: dict[int, int] = field(default_factory=dict)

    def season(self, start_year: int) -> list[Match]:
        return [m for m in self.matches if m.season == start_year]

    def played(self, before: datetime | None = None) -> list[Match]:
        return [m for m in self.matches if m.played and (before is None or m.date < before)]

    def teams(self, start_year: int) -> list[str]:
        s = self.season(start_year)
        return sorted({m.home for m in s} | {m.away for m in s})


SUPPLEMENTS_DIR = Path(__file__).resolve().parent / "supplements"


def apply_supplement(matches: list[Match], start_year: int) -> int:
    """Complète une saison avec les matchs d'un fichier de app/supplements.
    La source garde la priorité : on ne remplit que ce qui manque.
    Renvoie le nombre de matchs ajoutés ou complétés."""
    path = SUPPLEMENTS_DIR / f"top14-{start_year}-{start_year + 1}.json"
    if not path.exists():
        return 0
    sup = json.loads(path.read_text(encoding="utf-8"))
    by_pair = {(m.home, m.away): m for m in matches if not m.knockout}
    used = 0
    for s in sup["matches"]:
        m = by_pair.get((s["home"], s["away"]))
        if m is None:
            m = Match(
                season=start_year,
                date=datetime.fromisoformat(s["date"].replace("Z", "+00:00")),
                round=s.get("round"), home=s["home"], away=s["away"],
                knockout=False, neutral=False,
                reconstructed=True, date_approx=s.get("date_approx", False),
            )
            matches.append(m)
        elif m.played and m.has_detail:
            continue  # la source est complète pour ce match
        else:
            m.reconstructed = True
        if m.home_score is None:
            m.home_score, m.away_score = s["home_score"], s["away_score"]
        if m.home_tries is None or m.away_tries is None:
            m.home_tries, m.home_conv, m.home_kicks = s["home_tries"], s["home_conv"], s["home_kicks"]
            m.away_tries, m.away_conv, m.away_kicks = s["away_tries"], s["away_conv"], s["away_kicks"]
        if m.home_tries_any is None or m.away_tries_any is None:
            m.home_tries_any, m.away_tries_any = s["home_tries"], s["away_tries"]
        used += 1
    return used


def load_dataset(current: int | None = None) -> Dataset:
    current = current if current is not None else current_season_start()
    ds = Dataset(loaded_at=datetime.now(timezone.utc))
    files = dict(SEASON_FILES)
    for y in range(2026, current + 1):
        files.setdefault(y, f"top14-{y}-{y + 1}.json")
    for year, fname in sorted(files.items()):
        ttl = None if year < current else CACHE_TTL_CURRENT
        raw = _fetch_json(fname, ttl)
        if raw is None:
            ds.missing_seasons.append(year)
            continue
        season = parse_season(year, raw)
        n = apply_supplement(season, year)
        if n:
            ds.supplemented[year] = n
        ds.matches.extend(season)
    ds.matches.sort(key=lambda m: m.date)
    return ds
