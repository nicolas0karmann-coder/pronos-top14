"""
API du prédicteur Top 14.

Lancer :  uvicorn app.main:app --reload   (depuis le dossier backend)
puis ouvrir http://localhost:8000

En ligne (Render) : voir render.yaml à la racine du projet.
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import data as data_mod
from .model import ModelConfig, ScoreModel
from .season import simulate_season, standings

FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend"
REFRESH_SECONDS = 6 * 3600
MIN_FORCED_REFRESH_SECONDS = 10 * 60  # « Recharger les données » : au plus toutes les 10 min
log = logging.getLogger("top14")

# Réglages retenus après backtest (cf. README, section « Performances »)
MODEL_CONFIG = ModelConfig()



class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.loaded = 0.0
        self.ds: data_mod.Dataset | None = None
        self.season: int | None = None
        self.model: ScoreModel | None = None
        self.projection: dict | None = None
        self.round_models: dict[int, ScoreModel] = {}

    def ensure(self, force: bool = False):
        with self.lock:
            age = time.time() - self.loaded
            if self.ds is not None and age < (MIN_FORCED_REFRESH_SECONDS if force else REFRESH_SECONDS):
                return
            try:
                ds = data_mod.load_dataset()
                season = data_mod.current_season_start()
                if not ds.season(season):
                    raise RuntimeError("Calendrier de la saison en cours indisponible.")
                now = datetime.now(timezone.utc)
                model = ScoreModel(MODEL_CONFIG).fit(ds.matches, now, season, ds.teams(season))
            except Exception as e:  # source indisponible, données incomplètes...
                if self.ds is not None:
                    # on garde les données précédentes plutôt que de casser l'app
                    log.warning("Rafraîchissement impossible, données précédentes conservées : %s", e)
                    self.loaded = time.time() - REFRESH_SECONDS + 15 * 60  # nouvel essai dans 15 min
                    return
                raise HTTPException(503, f"Données indisponibles pour le moment : {e}")
            self.ds, self.season, self.model = ds, season, model
            self.projection = None
            self.round_models = {}
            self.loaded = time.time()

    def model_before_round(self, rnd: int) -> ScoreModel:
        """Modèle entraîné uniquement sur les matchs antérieurs à la journée
        (pour comparer honnêtement prédiction et résultat)."""
        with self.lock:
            if rnd in self.round_models:
                return self.round_models[rnd]
            matches = [m for m in self.ds.season(self.season) if m.round == rnd and not m.knockout]
            start = min(m.date for m in matches) - timedelta(hours=1)
            self.round_models[rnd] = ScoreModel(MODEL_CONFIG).fit(
                self.ds.matches, start, self.season, self.ds.teams(self.season)
            )
            return self.round_models[rnd]


state = State()


@asynccontextmanager
async def lifespan(_app):
    """Charge les données et entraîne le modèle dès le démarrage du serveur,
    en arrière-plan, pour que le premier visiteur n'attende pas."""
    def run():
        try:
            state.ensure()
            projection()
        except Exception as e:
            log.warning("Préchargement impossible : %s", e)
    threading.Thread(target=run, daemon=True).start()
    yield


app = FastAPI(title="Prédicteur Top 14", lifespan=lifespan)


@app.get("/api/health")
def health():
    """Vérification légère pour l'hébergeur (ne télécharge rien)."""
    return {"ok": True, "ready": state.ds is not None}


def _round_numbers():
    return sorted({m.round for m in state.ds.season(state.season) if not m.knockout and m.round})


def _current_round() -> int:
    """Première journée qui contient un match non joué."""
    season = [m for m in state.ds.season(state.season) if not m.knockout]
    pending = [m.round for m in season if not m.played]
    return min(pending) if pending else max(m.round for m in season)


def _round_payload(rnd: int) -> dict:
    matches = sorted(
        [m for m in state.ds.season(state.season) if m.round == rnd and not m.knockout],
        key=lambda m: m.date,
    )
    if not matches:
        raise HTTPException(404, f"Journée {rnd} introuvable.")
    any_played = any(m.played for m in matches)
    model = state.model_before_round(rnd) if any_played else state.model
    out = []
    correct = n_played = 0
    for m in matches:
        p = model.predict(m.home, m.away, m.neutral)
        item = {**m.to_dict(), "prediction": p}
        if m.played:
            n_played += 1
            fav = "home" if p["p_home"] >= max(p["p_draw"], p["p_away"]) else ("away" if p["p_away"] > p["p_draw"] else "draw")
            res = "home" if m.home_score > m.away_score else ("away" if m.away_score > m.home_score else "draw")
            item["prediction_correct"] = fav == res
            correct += fav == res
        out.append(item)
    return {
        "season": data_mod.season_label(state.season),
        "round": rnd,
        "rounds": _round_numbers(),
        "current_round": _current_round(),
        "matches": out,
        "played": n_played,
        "correct": correct,
        "model_trained_until": model.as_of.isoformat(),
    }


# ---------------------------------------------------------------- endpoints
@app.get("/api/statut")
def statut():
    state.ensure()
    ds = state.ds
    played = [m for m in ds.matches if m.played]
    warnings = []
    infos = []
    for year, n in sorted(ds.supplemented.items()):
        infos.append(
            f"Saison {data_mod.season_label(year)} : {n} matchs absents ou incomplets dans la source principale ont été "
            "complétés depuis la grille de résultats de Wikipédia. Leurs essais sont reconstitués à partir "
            "du score et des totaux officiels de la saison."
        )
    for year in {m.season for m in ds.matches}:
        if year >= state.season:
            continue
        n_league = sum(1 for m in ds.season(year) if m.played and not m.knockout)
        if year >= 2021 and n_league < 182:
            warnings.append(
                f"Saison {data_mod.season_label(year)} incomplète ({n_league} matchs sur 182) : "
                "la forme récente est moins précise."
            )
    if ds.missing_seasons:
        warnings.append("Saisons non téléchargées : " + ", ".join(map(str, ds.missing_seasons)))
    m = state.model
    return {
        "season": data_mod.season_label(state.season),
        "data_loaded_at": ds.loaded_at.isoformat(),
        "last_result": max(x.date for x in played).isoformat(),
        "matches_in_history": len(played),
        "matches_used_by_model": m.n_matches,
        "current_round": _current_round(),
        "home_advantage_points": m.global_home_advantage_points(),
        "conversion_rate": round(m.conv_rate, 3),
        "warnings": warnings,
        "infos": infos,
        "source": "github.com/transientlunatic/Rugby-Data (licence MIT), complétée par Wikipédia",
    }


@app.post("/api/rafraichir")
def rafraichir():
    state.ensure(force=True)
    return statut()


@app.get("/api/equipes")
def equipes():
    state.ensure()
    return {"season": data_mod.season_label(state.season), "teams": state.ds.teams(state.season)}


@app.get("/api/journee/courante")
def journee_courante():
    state.ensure()
    return _round_payload(_current_round())


@app.get("/api/journee/{rnd}")
def journee(rnd: int):
    state.ensure()
    return _round_payload(rnd)


@app.get("/api/predict")
def predict(home: str = Query(...), away: str = Query(...), neutre: bool = False):
    state.ensure()
    teams = state.ds.teams(state.season)
    for t in (home, away):
        if t not in teams:
            raise HTTPException(404, f"Équipe inconnue : {t}")
    if home == away:
        raise HTTPException(400, "Choisissez deux équipes différentes.")
    p = state.model.predict(home, away, neutre)
    # dernières confrontations
    h2h = [
        m.to_dict() for m in state.ds.matches
        if m.played and {m.home, m.away} == {home, away}
    ][-6:]
    return {**p, "head_to_head": list(reversed(h2h))}


@app.get("/api/classement")
def classement():
    state.ensure()
    season_matches = state.ds.season(state.season)
    return {
        "season": data_mod.season_label(state.season),
        "rows": standings(season_matches, state.ds.teams(state.season)),
    }


@app.get("/api/projection")
def projection():
    state.ensure()
    if state.projection is None:
        state.projection = simulate_season(
            state.model, state.ds.season(state.season), state.ds.teams(state.season), n_sims=10000
        )
    return {"season": data_mod.season_label(state.season), **state.projection}


@app.get("/api/forces")
def forces():
    state.ensure()
    return {
        "season": data_mod.season_label(state.season),
        "home_advantage_points": state.model.global_home_advantage_points(),
        "rows": state.model.ratings(state.ds.teams(state.season)),
    }


@app.get("/api/equipe/{team}")
def equipe(team: str):
    state.ensure()
    if team not in state.ds.teams(state.season):
        raise HTTPException(404, f"Équipe inconnue : {team}")
    season = state.ds.season(state.season)
    played = [m.to_dict() for m in season if m.played and team in (m.home, m.away)]
    upcoming = []
    for m in season:
        if not m.played and not m.knockout and team in (m.home, m.away):
            p = state.model.predict(m.home, m.away, m.neutral)
            upcoming.append({**m.to_dict(), "prediction": p})
    return {"team": team, "played": played, "upcoming": upcoming[:5]}


# ---------------------------------------------------------------- frontend
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND / "index.html")
