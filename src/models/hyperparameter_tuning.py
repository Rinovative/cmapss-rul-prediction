import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV

from .models import nasa_score

nasa_scorer = make_scorer(nasa_score, greater_is_better=False)


def run_grid_search(
    X_train,
    y_train,
    X_test,
    y_test,
    models,  # list[(name, estimator, supports_weight)]
    param_grids: dict,
    model_path: str | None = None,
    force_run: bool = False,
) -> pd.DataFrame:
    """
    Grid-Search CV mit optionalem Laden/Speichern.

    Wird ein bereits gespeichertes Best-Modell samt Ergebnis-DataFrame
    gefunden und *force_run=False*, werden beide geladen.
    Andernfalls wird GridSearchCV neu ausgeführt, das beste Modell auf
    den kompletten Trainingsdaten nachtrainiert und beides gespeichert.

    ----------
    Parameters
    ----------
    X_train, y_train, X_test, y_test
        Trainings- und Testdaten (Features / Zielvariable).
    models : list[tuple[str, estimator, bool]]
        • Name  – Klartext­bezeichnung
        • estimator – **ungefittete** sklearn-Instanz
        • bool – True, falls ``sample_weight`` unterstützt wird.
    param_grids : dict[str, dict]
        Hyperparameter-Gitter je Modell­name.
    model_path : str | None, default=None
        Dateipfad für das beste Modell (*.pkl*).
        Wird ``None`` gesetzt, erfolgt kein Laden/Speichern.
        Das zugehörige Ergebnis-Pickle wird unter
        ``model_path.replace('.pkl', '_grid.pkl')`` abgelegt bzw. geladen.
    force_run : bool, default=False
        • False  → Versucht zuerst zu laden, falls Dateien existieren.
        • True   → Berechnet GridSearch immer neu und überschreibt Dateien.

    ----------
    Returns
    -------
    pd.DataFrame
        Tabelle mit Spalten::

            Model          – Modellname
            Best Params    – bestes Hyperparameter-Dict
            RMSE-Test      – RMSE auf Testdaten
            R²-Test        – R² auf Testdaten
            NASA-Score     – NASA-Score auf Testdaten

        sortiert aufsteigend nach *NASA-Score*.
    """
    # Ablage für Grid-Result-DF
    df_path = None
    if model_path:
        df_path = model_path.replace(".pkl", "_grid.pkl")

    # -------------------------- 0) Laden, falls vorhanden --------------------
    if model_path and df_path and not force_run and os.path.exists(model_path) and os.path.exists(df_path):
        print(f"🔄  Lade Modell & Grid-DF aus {model_path}")
        return pd.read_pickle(df_path)

    # -------------------------- 1) GridSearch durchführen --------------------
    results = []
    for name, est, supports in models:
        grid = GridSearchCV(est, param_grids.get(name, {}), scoring=nasa_scorer, cv=3, n_jobs=-1)

        if supports:
            weights = 1 + 2 * np.exp(-y_train / 25)
            grid.fit(X_train, y_train, sample_weight=weights)
        else:
            grid.fit(X_train, y_train)

        best = grid.best_estimator_
        y_pred = best.predict(X_test)

        results.append((name, grid.best_params_, root_mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), nasa_score(y_test, y_pred)))

    df = (
        pd.DataFrame(results, columns=["Model", "Best Params", "RMSE-Test", "R²-Test", "NASA-Score"]).sort_values("NASA-Score").reset_index(drop=True)
    )

    # -------------------------- 2) Bestes Modell nachtrainieren --------------
    if model_path:
        best_name = df.iloc[0]["Model"]

        # Tuple aus models-Liste heraussuchen
        base_est, supports = next((clone(est), sw) for (n, est, sw) in models if n == best_name)

        if supports:
            base_est.fit(X_train, y_train, sample_weight=1 + 2 * np.exp(-y_train / 25))
        else:
            base_est.fit(X_train, y_train)

        # speichern
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(base_est, model_path)
        df.to_pickle(df_path)
        print(f"💾  Neues Modell & DF gespeichert ({model_path})")

    return df


def run_random_search(
    X_train,
    y_train,
    X_test,
    y_test,
    groups,  # Array für GroupKFold
    models,  # list[(name, estimator, supports_weight)]
    param_distributions: dict[str, dict],
    n_splits: int = 3,
    n_iter: int = 200,
    model_path: str | None = None,
    force_run: bool = False,
) -> pd.DataFrame:
    """
    RandomizedSearchCV mit automatischem Laden/Speichern.

    Lädt vorhandenes Modell + Ergebnis-DF, sofern beide Dateien existieren
    und `force_run=False`.  Ansonsten wird RandomizedSearchCV nur für
    diejenigen Modelle durchgeführt, die einen Eintrag in
    `param_distributions` haben.  Das beste Modell wird anschließend auf
    den vollen Trainingsdaten nachtrainiert und zusammen mit dem Ergebnis-
    DataFrame gespeichert.

    Parameters
    ----------
    models : list[tuple[str, estimator, bool]]
        Name, ungefittete Instanz, supports_weight-Flag.
    param_distributions : dict[str, dict]
        Nur Modelle mit einem Eintrag werden getunt.
    ...

    Returns
    -------
    pd.DataFrame
        Tabelle mit NASA-Score, RMSE, R² und besten Parametern.
    """
    # ---------- 0) Laden, falls möglich ------------------------------------
    df_path = model_path.replace(".pkl", "_rand.pkl") if model_path else None
    if model_path and df_path and not force_run and os.path.exists(model_path) and os.path.exists(df_path):
        print(f"🔄  Lade RandomSearch-Ergebnisse und Modell aus {model_path}")
        return pd.read_pickle(df_path)

    # ---------- 1) RandomizedSearchCV --------------------------------------
    gkf = GroupKFold(n_splits=n_splits)
    results = []

    for name, estimator, supports_weight in models:
        # --- nur Modelle mit Parameterräumen tunen
        param_dist = param_distributions.get(name)
        if not param_dist:
            continue  # überspringen

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=nasa_scorer,
            cv=gkf.split(X_train, y_train, groups=groups),
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        if supports_weight:
            weights = 1 + 2 * np.exp(-y_train / 25)
            search.fit(X_train, y_train, sample_weight=weights)
        else:
            search.fit(X_train, y_train)

        best = search.best_estimator_
        y_pred = best.predict(X_test)

        results.append((name, search.best_params_, root_mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), nasa_score(y_test, y_pred)))

    df = (
        pd.DataFrame(results, columns=["Model", "Best Params", "RMSE-Test", "R²-Test", "NASA-Score"]).sort_values("NASA-Score").reset_index(drop=True)
    )

    # ---------- 2) Bestes Modell nachtrainieren + speichern ---------------
    if model_path and not df.empty:
        best_name = df.iloc[0]["Model"]
        best_est, supports = next((clone(est), sw) for (n, est, sw) in models if n == best_name)

        if supports:
            best_est.fit(X_train, y_train, sample_weight=1 + 2 * np.exp(-y_train / 25))
        else:
            best_est.fit(X_train, y_train)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_est, model_path)
        df.to_pickle(df_path)
        print(f"💾  RandomSearch-Modell + Ergebnis gespeichert → {model_path}")

    return df


def select_best_per_model(df_grid, df_rand):
    """
    Vergleicht pro Modell die NASA-Scores von GridSearch (df_grid) und RandomSearch (df_rand)
    und übernimmt den besseren. Falls RandomSearch schlechter ist, bleibt nur GridSearch.
    """
    combined_rows = []

    for model in df_grid["Model"].unique():
        row_grid = df_grid[df_grid["Model"] == model].iloc[0]
        row_rand_match = df_rand[df_rand["Model"] == model]

        if not row_rand_match.empty:
            row_rand = row_rand_match.iloc[0]
            if row_rand["NASA-Score"] < row_grid["NASA-Score"]:
                combined_rows.append(row_rand)
            else:
                # RandomSearch war schlechter → trotzdem Grid übernehmen
                combined_rows.append(row_grid)
        else:
            # kein RandomSearch vorhanden → Grid übernehmen
            combined_rows.append(row_grid)

    return pd.DataFrame(combined_rows).reset_index(drop=True)


def expand_best_params(df):
    """
    Entpackt die Dictionaries in der Spalte "Best Params" in einzelne Spalten,
    sodass jede Hyperparameter-Spalte separat im DataFrame erscheint.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame, das eine Spalte "Best Params" enthält.
        Jede Zelle dieser Spalte ist ein dict mit Hyperparameter-Namen als Schlüssel
        und deren besten Werten als Werte.

    Returns
    -------
    pd.DataFrame
        Neues DataFrame, in dem die Spalte "Best Params" entfernt wurde und stattdessen
        für jeden Hyperparameter eine eigene Spalte existiert. Andere Spalten bleiben erhalten.
    """
    params_expanded = df["Best Params"].apply(pd.Series)
    df_expanded = pd.concat([df.drop(columns=["Best Params"]), params_expanded], axis=1)
    return df_expanded
