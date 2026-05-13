"""
Génération du fichier dashboard_data.json pour le projet
Québec City Permit Intelligence Dashboard.

Ce script :
1. charge le fichier CSV des permis délivrés à la Ville de Québec ;
2. nettoie et prépare les données ;
3. protège les informations sensibles en excluant les adresses, les lots et les numéros de permis du JSON ;
4. agrège les données pour la carte interactive ;
5. produit des indicateurs analytiques ;
6. entraîne, lorsque les données sont suffisantes, des modèles de classification ;
7. entraîne, lorsque les données sont suffisantes, des modèles de prévision du volume mensuel ;
8. génère dashboard_data.json pour alimenter le dashboard web.

Exécution :
    python build_dashboard_data.py

Fichier attendu :
    data/vdq-permis.csv
ou :
    vdq-permis.csv

Fichier généré :
    dashboard_data.json
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

PROJECT_NAME = "Québec City Permit Intelligence Dashboard"
ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [ROOT / "data" / "vdq-permis.csv", ROOT / "vdq-permis.csv"]
OUTPUT_JSON = ROOT / "dashboard_data.json"

REQUIRED_COLUMNS = [
    "NUMERO_PERMIS",
    "DATE_DELIVRANCE",
    "ADRESSE_TRAVAUX",
    "DOMAINE",
    "LOTS_IMPACTES",
    "TYPE_PERMIS",
    "ARRONDISSEMENT",
    "RAISON",
    "LONGITUDE",
    "LATITUDE",
]

SENSITIVE_COLUMNS = ["NUMERO_PERMIS", "ADRESSE_TRAVAUX", "LOTS_IMPACTES", "RAISON"]


def log(message: str) -> None:
    print(f"[build] {message}")


def find_data_file() -> Path:
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Aucun fichier CSV trouvé. Placez le fichier 'vdq-permis.csv' dans le dossier du projet "
        "ou dans le dossier 'data/'."
    )


def read_csv_safely(path: Path) -> pd.DataFrame:
    for encoding in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return "Non précisé"
    text = str(value).strip()
    return text if text else "Non précisé"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    data = df.copy()
    data["DATE_DELIVRANCE"] = pd.to_datetime(data["DATE_DELIVRANCE"], errors="coerce")
    data["LONGITUDE"] = pd.to_numeric(data["LONGITUDE"], errors="coerce")
    data["LATITUDE"] = pd.to_numeric(data["LATITUDE"], errors="coerce")

    for col in ["DOMAINE", "TYPE_PERMIS", "ARRONDISSEMENT"]:
        data[col] = data[col].map(normalize_text)

    data = data.dropna(subset=["DATE_DELIVRANCE", "LONGITUDE", "LATITUDE"])

    data["annee"] = data["DATE_DELIVRANCE"].dt.year.astype(int)
    data["mois"] = data["DATE_DELIVRANCE"].dt.month.astype(int)
    data["annee_mois"] = data["DATE_DELIVRANCE"].dt.to_period("M").astype(str)
    data["jour_semaine"] = data["DATE_DELIVRANCE"].dt.day_name(locale=None)
    data["trimestre"] = data["DATE_DELIVRANCE"].dt.quarter.astype(int)

    # Protection de la précision spatiale : coordonnées arrondies.
    # 3 décimales correspondent approximativement à une maille d'environ 100 m.
    data["lat_grid"] = data["LATITUDE"].round(3)
    data["lon_grid"] = data["LONGITUDE"].round(3)

    return data


def top_counts(df: pd.DataFrame, column: str, limit: int = 10) -> list[dict]:
    counts = df[column].value_counts(dropna=False).head(limit)
    total = len(df)
    return [
        {"label": str(idx), "count": int(value), "share": round(float(value / total * 100), 2) if total else 0}
        for idx, value in counts.items()
    ]


def build_map_points(df: pd.DataFrame) -> list[dict]:
    grouped = (
        df.groupby(["lat_grid", "lon_grid", "ARRONDISSEMENT", "DOMAINE", "TYPE_PERMIS", "annee", "mois"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    records = []
    for row in grouped.itertuples(index=False):
        records.append(
            {
                "lat": float(row.lat_grid),
                "lon": float(row.lon_grid),
                "arrondissement": row.ARRONDISSEMENT,
                "domaine": row.DOMAINE,
                "type_permis": row.TYPE_PERMIS,
                "annee": int(row.annee),
                "mois": int(row.mois),
                "count": int(row.count),
            }
        )
    return records


def build_monthly_series(df: pd.DataFrame) -> list[dict]:
    series = df.groupby("annee_mois").size().reset_index(name="count").sort_values("annee_mois")
    return [{"period": row.annee_mois, "count": int(row.count)} for row in series.itertuples(index=False)]


def build_monthly_by_arrondissement(df: pd.DataFrame) -> list[dict]:
    series = (
        df.groupby(["annee_mois", "ARRONDISSEMENT"])
        .size()
        .reset_index(name="count")
        .sort_values(["annee_mois", "ARRONDISSEMENT"])
    )
    return [
        {"period": row.annee_mois, "arrondissement": row.ARRONDISSEMENT, "count": int(row.count)}
        for row in series.itertuples(index=False)
    ]


def safe_div(a: float, b: float) -> float:
    if not b:
        return 0.0
    return float(a / b)


def train_classification(df: pd.DataFrame) -> dict:
    # Objectif : prédire TYPE_PERMIS à partir de variables non sensibles.
    # On évite NUMERO_PERMIS, ADRESSE_TRAVAUX, LOTS_IMPACTES et RAISON.
    modeling = df[["TYPE_PERMIS", "DOMAINE", "ARRONDISSEMENT", "annee", "mois", "trimestre", "lat_grid", "lon_grid"]].copy()
    class_counts = modeling["TYPE_PERMIS"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    modeling = modeling[modeling["TYPE_PERMIS"].isin(valid_classes)]

    if len(modeling) < 50 or modeling["TYPE_PERMIS"].nunique() < 2:
        return {
            "status": "insufficient_data",
            "message": "Données insuffisantes pour entraîner un modèle de classification fiable sur cet échantillon.",
            "target": "TYPE_PERMIS",
            "models": [],
            "best_model": None,
        }

    X = modeling.drop(columns=["TYPE_PERMIS"])
    y = modeling["TYPE_PERMIS"]

    cat_features = ["DOMAINE", "ARRONDISSEMENT"]
    num_features = ["annee", "mois", "trimestre", "lat_grid", "lon_grid"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features),
        ]
    )

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    models = {
        "Régression logistique équilibrée": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced"),
        "Extra Trees": ExtraTreesClassifier(n_estimators=160, random_state=42, class_weight="balanced"),
    }

    results = []
    best_name = None
    best_score = -1.0

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = f1_score(y_test, pred, average="macro", zero_division=0)
        results.append({"model": name, "metric": "f1_macro", "score": round(float(score), 4)})
        if score > best_score:
            best_score = score
            best_name = name

    return {
        "status": "trained",
        "target": "TYPE_PERMIS",
        "metric": "f1_macro",
        "best_model": best_name,
        "models": sorted(results, key=lambda item: item["score"], reverse=True),
        "note": "Classification réalisée sans utiliser les colonnes sensibles : numéro de permis, adresse, lots et raison détaillée.",
    }


def train_forecast(df: pd.DataFrame) -> dict:
    monthly = df.groupby("annee_mois").size().reset_index(name="count").sort_values("annee_mois")
    monthly["date"] = pd.to_datetime(monthly["annee_mois"] + "-01")
    monthly["month_index"] = np.arange(len(monthly))
    monthly["annee"] = monthly["date"].dt.year
    monthly["mois"] = monthly["date"].dt.month
    monthly["trimestre"] = monthly["date"].dt.quarter

    if len(monthly) < 8:
        return {
            "status": "insufficient_data",
            "message": "Historique mensuel insuffisant pour entraîner un modèle de prévision fiable sur cet échantillon.",
            "models": [],
            "best_model": None,
            "forecast": [],
        }

    features = ["month_index", "annee", "mois", "trimestre"]
    X = monthly[features]
    y = monthly["count"]

    test_size = max(2, math.ceil(len(monthly) * 0.25))
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    models = {
        "Régression linéaire": LinearRegression(),
        "Forêt aléatoire": RandomForestRegressor(n_estimators=180, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=180, random_state=42),
    }

    results = []
    best_name = None
    best_mae = float("inf")
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        results.append({"model": name, "metric": "mae", "score": round(float(mae), 3)})
        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_model = model

    assert best_model is not None
    best_model.fit(X, y)

    last_date = monthly["date"].max()
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=6, freq="MS")
    future = pd.DataFrame(
        {
            "date": future_dates,
            "month_index": np.arange(len(monthly), len(monthly) + len(future_dates)),
        }
    )
    future["annee"] = future["date"].dt.year
    future["mois"] = future["date"].dt.month
    future["trimestre"] = future["date"].dt.quarter
    future_pred = np.maximum(0, best_model.predict(future[features]))

    forecast = [
        {"period": date.strftime("%Y-%m"), "predicted_count": int(round(value))}
        for date, value in zip(future["date"], future_pred)
    ]

    return {
        "status": "trained",
        "target": "volume_mensuel_permis",
        "metric": "mae",
        "best_model": best_name,
        "models": sorted(results, key=lambda item: item["score"]),
        "forecast": forecast,
        "note": "Prévision du volume mensuel global de permis délivrés.",
    }


def build_dashboard_data(df: pd.DataFrame, source_file: Path) -> dict:
    min_date = df["DATE_DELIVRANCE"].min()
    max_date = df["DATE_DELIVRANCE"].max()

    options = {
        "annees": sorted([int(x) for x in df["annee"].dropna().unique()]),
        "mois": sorted([int(x) for x in df["mois"].dropna().unique()]),
        "arrondissements": sorted(df["ARRONDISSEMENT"].dropna().astype(str).unique().tolist()),
        "domaines": sorted(df["DOMAINE"].dropna().astype(str).unique().tolist()),
        "types_permis": sorted(df["TYPE_PERMIS"].dropna().astype(str).unique().tolist()),
    }

    dashboard_data = {
        "metadata": {
            "project_name": PROJECT_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source_file.name,
            "rows_after_cleaning": int(len(df)),
            "date_min": min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else None,
            "date_max": max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else None,
            "licence_donnees": "Attribution Creative Commons 4.0 International (CC BY 4.0)",
            "source_description": "Permis délivrés à la Ville de Québec — portail Données Québec.",
            "privacy_note": "Le JSON exclut les numéros de permis, les adresses exactes, les lots impactés et les raisons détaillées. Les coordonnées utilisées pour la carte sont arrondies afin de réduire la précision spatiale.",
        },
        "columns_used": {
            "kept_for_analysis": [
                "DATE_DELIVRANCE",
                "DOMAINE",
                "TYPE_PERMIS",
                "ARRONDISSEMENT",
                "LONGITUDE",
                "LATITUDE",
            ],
            "excluded_from_dashboard_json": SENSITIVE_COLUMNS,
        },
        "summary": {
            "total_permis": int(len(df)),
            "total_arrondissements": int(df["ARRONDISSEMENT"].nunique()),
            "total_domaines": int(df["DOMAINE"].nunique()),
            "total_types_permis": int(df["TYPE_PERMIS"].nunique()),
            "top_arrondissements": top_counts(df, "ARRONDISSEMENT", 10),
            "top_domaines": top_counts(df, "DOMAINE", 10),
            "top_types_permis": top_counts(df, "TYPE_PERMIS", 10),
        },
        "options": options,
        "time_series": {
            "monthly_total": build_monthly_series(df),
            "monthly_by_arrondissement": build_monthly_by_arrondissement(df),
        },
        "map": {
            "center": {
                "lat": round(float(df["LATITUDE"].mean()), 5) if len(df) else 46.8139,
                "lon": round(float(df["LONGITUDE"].mean()), 5) if len(df) else -71.2080,
            },
            "points": build_map_points(df),
        },
        "models": {
            "classification_type_permis": train_classification(df),
            "forecast_volume_mensuel": train_forecast(df),
        },
    }
    return dashboard_data


def main() -> None:
    data_file = find_data_file()
    log(f"Chargement du fichier : {data_file}")
    raw = read_csv_safely(data_file)
    log(f"Lignes chargées : {len(raw):,}".replace(",", " "))

    cleaned = clean_data(raw)
    log(f"Lignes conservées après nettoyage : {len(cleaned):,}".replace(",", " "))
    log("Construction du fichier dashboard_data.json")

    dashboard_data = build_dashboard_data(cleaned, data_file)

    OUTPUT_JSON.write_text(json.dumps(dashboard_data, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Fichier généré : {OUTPUT_JSON.name} ({OUTPUT_JSON.stat().st_size / (1024 * 1024):.2f} Mo)")
    log("Terminé.")


if __name__ == "__main__":
    main()
