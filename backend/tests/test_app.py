"""Tests de bout en bout (nécessitent l'accès à GitHub ou un cache rempli)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.data import load_dataset
from app.main import app
from app.model import ScoreModel
from app.season import league_points, standings

client = TestClient(app)


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def test_bareme_top14():
    # victoire avec 3 essais de plus : 4 + bonus offensif ; perdant à 5 pts : bonus défensif
    assert league_points(30, 25, 5, 2)[:2] == (5, 1)
    # victoire avec seulement 2 essais de plus : pas de bonus offensif
    assert league_points(30, 10, 4, 2)[:2] == (4, 0)
    # défaite de 6 points : pas de bonus défensif
    assert league_points(20, 26, 2, 2)[:2] == (0, 4)
    assert league_points(20, 20, 2, 2)[:2] == (2, 2)


def test_classement_2024_25_officiel(ds):
    """Le classement recalculé doit correspondre au classement officiel 2024-25."""
    official = {"Toulouse": 90, "Bordeaux-Bègles": 78, "Toulon": 72, "Bayonne": 68, "Clermont": 63,
                "Castres": 63, "La Rochelle": 62, "Pau": 61, "Montpellier": 56, "Racing 92": 56,
                "Lyon": 50, "Stade Français": 45, "Perpignan": 44, "Vannes": 36}
    rows = standings(ds.season(2024), ds.teams(2024))
    assert {r["team"]: r["points"] for r in rows} == official


def test_probabilites_coherentes(ds):
    from datetime import datetime, timezone
    m = ScoreModel().fit(ds.matches, datetime(2025, 1, 1, tzinfo=timezone.utc), 2024, ds.teams(2024))
    p = m.predict("Toulouse", "Perpignan")
    assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1) < 1e-6
    assert p["p_home"] > p["p_away"]
    assert abs(sum(p["margin_buckets"].values()) - 1) < 1e-6
    # avantage du terrain : en inversant l affiche, Perpignan doit y gagner
    q = m.predict("Perpignan", "Toulouse")
    assert q["p_home"] > p["p_away"]  # Perpignan a plus de chances chez lui
    # un bonus offensif implique une victoire
    assert p["bonus"]["offensif_home"] <= p["p_home"]


def test_simulation_coherente(ds):
    r = client.get("/api/projection").json()
    teams = r["teams"]
    assert abs(sum(t["p_first"] for t in teams) - 1) < 1e-6
    assert abs(sum(t["p_top6"] for t in teams) - 6) < 1e-6
    for t in teams:
        assert abs(sum(t["rank_distribution"]) - 1) < 1e-3
        assert t["expected_points"] >= t["current_points"]


@pytest.mark.parametrize("url", [
    "/api/statut", "/api/journee/courante", "/api/journee/1", "/api/classement",
    "/api/forces", "/api/equipes", "/api/predict?home=Toulouse&away=Pau",
])
def test_endpoints(url):
    assert client.get(url).status_code == 200


def test_erreurs():
    assert client.get("/api/predict?home=Toulouse&away=Toulouse").status_code == 400
    assert client.get("/api/predict?home=Toulouse&away=Inconnu").status_code == 404
    assert client.get("/api/journee/99").status_code == 404


def test_index():
    r = client.get("/")
    assert r.status_code == 200 and "Pronos Top 14" in r.text


def test_complement_2025_26(ds):
    """La saison 2025-26 complétée doit compter ses 182 matchs, et le classement
    recalculé doit retrouver les points officiels (hors Toulouse, Montpellier
    et Lyon : sanction ou bonus incertain, cf. README)."""
    s = [m for m in ds.season(2025) if m.played and not m.knockout]
    assert len(s) == 182
    official = {"Stade Français": 79, "Pau": 78, "Racing 92": 74, "La Rochelle": 72, "Clermont": 71,
                "Bordeaux-Bègles": 70, "Toulon": 59, "Castres": 55, "Bayonne": 51, "Perpignan": 29,
                "Montauban": 7}
    pts = {r["team"]: r["points"] for r in standings(ds.season(2025), ds.teams(2025))}
    assert {k: pts[k] for k in official} == official
