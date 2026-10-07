# Pronos Top 14

Web app de prédiction pour le Top 14 : pronostics de chaque journée (victoire, nul, score, essais, bonus), classement actuel, projection de fin de saison et comparateur de matchs.

## Lancer l'app

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Puis ouvrir http://localhost:8000.

Au premier lancement, l'app télécharge environ 13 saisons depuis GitHub (quelques secondes) et les garde dans `backend/cache/`. Les saisons terminées ne sont plus retéléchargées. La saison en cours est rafraîchie toutes les 6 heures, ou à la demande avec « Recharger les données » en bas de page.

## Mettre l'app en ligne (Render, offre gratuite)

Le projet contient un fichier `render.yaml` qui décrit tout le service. Il n'y a rien à régler à la main.

1. **GitHub** : créez un nouveau dépôt (par exemple `pronos-top14`), puis « Add file » > « Upload files ». Glissez-y **le contenu** du dossier `top14_predictor` : `render.yaml`, `README.md`, `.gitignore` et les dossiers `backend` et `frontend`. `render.yaml` doit être à la racine du dépôt.
2. **Render** : « New » > « Blueprint », choisissez le dépôt, puis « Apply ». Render installe les dépendances et lance le serveur, en 3 à 5 minutes la première fois.
3. L'adresse publique s'affiche sur la page du service (du type `https://pronos-top14.onrender.com`). C'est la web app complète : le site et l'API sont servis ensemble.

Ensuite, chaque modification envoyée sur GitHub redéploie automatiquement l'app.

À savoir sur l'offre gratuite :
- Le serveur s'endort après 15 minutes sans visite. La visite suivante le réveille, ce qui prend environ une minute. Il retélécharge alors les données ; un message l'indique sur la page pendant l'attente.
- Le processeur est limité. Afficher une journée déjà jouée réentraîne le modèle à la date de cette journée, ce qui peut prendre quelques secondes la première fois (le résultat est ensuite gardé en mémoire).
- Aucune clé ni variable secrète n'est nécessaire : les données viennent d'un dépôt GitHub public.

## Ce que fait l'app

- **Journée** : pour chaque match, les probabilités de victoire (barre en forme de terrain), le score moyen prévu, les essais attendus, les chances de bonus offensif et défensif, et les points de classement attendus. Pour une journée déjà jouée, la prédiction affichée est celle d'un modèle entraîné *avant* la journée, avec le résultat réel en face.
- **Classement** : recalculé à partir des résultats avec le barème officiel.
- **Fin de saison** : 10 000 simulations des matchs restants, bonus compris. On obtient les points finaux prévus, les probabilités de top 2, top 6, 13e et 14e place, et la répartition des places.
- **Match au choix** : n'importe quelle affiche, éventuellement sur terrain neutre (utile pour les demi-finales et la finale), avec la répartition des écarts et les dernières confrontations.
- **Forces** : un indice par équipe (attaque, défense, avantage du terrain propre au club).

## Le modèle

Le Dixon-Coles de l'app foot ne convient pas au rugby : les points arrivent par paquets de 3, 5 et 7, donc une loi de Poisson sur le score n'a pas de sens. Le modèle sépare donc les deux façons de marquer.

1. **Essais** et **coups de pied réussis** (pénalités + drops) sont chacun modélisés par une loi de Poisson, avec une attaque et une défense par équipe, un avantage du terrain global et un avantage propre à chaque club.
2. Chaque essai est transformé avec une probabilité estimée sur les données (environ 77 %).
3. Deux dépendances observées dans les données sont prises en compte :
   - une équipe qui marque beaucoup d'essais passe moins de pénalités (corrélation de −0,43) ;
   - les essais des deux équipes montent ensemble dans les matchs ouverts (loi de Poisson bivariée).

   Sans elles, le modèle surestimait nettement la dispersion des écarts.
4. Les matchs anciens comptent moins (demi-vie de 270 jours). Le niveau général du championnat (scores moyens, avantage du terrain) est recalé sur une fenêtre plus courte (90 jours), parce que les scores du Top 14 augmentent d'une saison à l'autre.
5. Les forces sont tirées vers la moyenne (pénalité ridge) pour éviter les valeurs extrêmes sur peu de matchs. Les promus reçoivent un handicap appris sur tous les promus de l'historique.

On reconstitue ensuite exactement la loi du score de chaque équipe, d'où les probabilités de résultat et de bonus. Le bonus offensif dépend de l'écart d'essais, le bonus défensif de l'écart de points.

Barème utilisé : victoire 4, nul 2. Bonus offensif pour une victoire avec au moins 3 essais de plus que l'adversaire, bonus défensif pour une défaite de 5 points ou moins.

## Performances (backtest)

Pour chaque journée des saisons 2022-23 à 2026-27, le modèle est entraîné uniquement sur les matchs joués avant, puis comparé aux résultats. Cela représente 693 matchs de saison régulière. Pour la log-loss, plus bas = mieux.

| Méthode | Log-loss | Brier | Bon vainqueur | Erreur moyenne sur l'écart |
|---|---|---|---|---|
| Fréquences moyennes (toujours ~76 % domicile) | 0,628 | 0,376 | 75,9 % | 12,1 pts |
| Elo en points d'écart | 0,610 | 0,362 | 76,5 % | 11,2 pts |
| **Modèle de l'app** | **0,598** | **0,353** | **77,2 %** | **11,0 pts** |

Relancer le backtest : `python -m app.backtest` (depuis `backend`, environ 30 secondes).

Les réglages (demi-vie, rétrécissement) ont été choisis sur ce même backtest. Les gains entre réglages voisins étant minimes, le risque de sur-ajustement reste faible, mais il existe.

## Limites connues

- **Saison 2025-26 incomplète** : la source s'arrête mi-février 2026 (112 matchs sur 187). La forme de fin de saison dernière manque donc, ce qui pèse surtout sur les premières journées de 2026-27. L'app l'affiche en bandeau.
- **Saisons absentes de la source** : 2016-17, 2019-20 et 2020-21 (Covid). L'historique détaillé (essais) commence en 2014-15.
- **Biais résiduel** : en backtest, l'équipe qui reçoit marque en moyenne 1,2 point de plus que prévu. Les victoires à domicile entre 60 et 80 % de probabilité prévue arrivent en réalité un peu plus souvent (77 % observé). Les bonus défensifs sont sous-estimés (12 % prévu contre 16 % observé).
- **Phases finales** : la projection s'arrête à la fin de la saison régulière. Les barrages, demi-finales et finale ne sont pas simulés, mais l'onglet « Match au choix » permet de pronostiquer une affiche sur terrain neutre.
- **Départage au classement** : points, puis différence de points, puis essais. Le règlement officiel commence par les confrontations directes ; l'ordre peut donc différer en cas d'égalité de points.
- Le modèle ne connaît ni les compositions, ni les blessures, ni les doublons internationaux, ni les matchs où un club fait tourner son effectif.

## Données

[transientlunatic/Rugby-Data](https://github.com/transientlunatic/Rugby-Data) (licence MIT), mis à jour automatiquement chaque lundi. Corrections appliquées au chargement (`app/data.py`) :
- les deux premiers fichiers sont mal nommés ;
- avant 2021, l'année des dates est fausse ;
- un même club apparaît sous plusieurs noms selon les saisons ;
- les phases finales ne sont pas toujours marquées.

## Structure

```
render.yaml      configuration Render
backend/
  requirements.txt      dépendances du serveur
  requirements-dev.txt  + outils de test
  app/
    data.py      chargement, nettoyage, cache
    model.py     modèle essais + coups de pied
    season.py    classement et simulation de fin de saison
    backtest.py  évaluation à l'aveugle
    main.py      API FastAPI (sert aussi le front)
  tests/         tests de bout en bout (pytest)
frontend/        index.html, style.css, app.js
```

## API

| Route | Contenu |
|---|---|
| `GET /api/journee/courante` | Prochaine journée avec pronostics |
| `GET /api/journee/{n}` | Journée n (prédiction d'avant-match + résultat si joué) |
| `GET /api/predict?home=…&away=…&neutre=false` | Pronostic d'une affiche quelconque |
| `GET /api/classement` | Classement actuel |
| `GET /api/projection` | Simulation de fin de saison |
| `GET /api/forces` | Indices de force |
| `GET /api/equipe/{nom}` | Matchs joués et 5 prochains pronostics d'un club |
| `GET /api/statut` | État des données et avertissements |
| `POST /api/rafraichir` | Retélécharge la saison en cours (au plus toutes les 10 min) |
| `GET /api/health` | Vérification de l'hébergeur |
