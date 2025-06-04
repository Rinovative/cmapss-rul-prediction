import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV


def nasa_score(y_true, y_pred):
    """
    Berechnet den NASA-Score zur Bewertung von RUL-Vorhersagen (Remaining Useful Life).
    Der Score bestraft Über- und Unterschätzungen exponentiell unterschiedlich:
      - Unterschätzung (delta < 0): Exponentialstrafe mit Basis e^(-delta/13) − 1
      - Überschätzung (delta > 0): Exponentialstrafe mit Basis e^(delta/10) − 1

    Parameters
    ----------
    y_true : array-like, Form (n_samples,)
        Die tatsächlichen RUL-Werte (Remaining Useful Life).
    y_pred : array-like, Form (n_samples,)
        Die vorhergesagten RUL-Werte.

    Returns
    -------
    float
        Der aufsummierte NASA-Score über alle Beispiele.
        Kleinere Werte sind besser (bessere Vorhersage), da greater_is_better=False.
    """
    delta = y_pred - y_true
    score = np.where(delta < 0, np.exp(-delta / 13) - 1, np.exp(delta / 10) - 1)
    return np.sum(score)


nasa_scorer = make_scorer(nasa_score, greater_is_better=False)


def run_grid_search(X_train, y_train, X_test, y_test, models, param_grids):
    """
    Führt für eine gegebene Liste von Modellen jeweils eine GridSearchCV aus und ermittelt
    die besten Hyperparameter gemäß dem NASA-Score. Anschließend werden für jeden besten
    Modell-Parameter die Performance-Kennzahlen (RMSE, R², NASA-Score) auf den Testdaten berechnet.

    Parameters
    ----------
    X_train : pd.DataFrame oder np.ndarray, Form (n_train_samples, n_features)
        Trainingsmerkmale (Features) ohne die Spalte 'unit', falls vorhanden.
    y_train : pd.Series oder np.ndarray, Form (n_train_samples,)
        Die tatsächlichen RUL-Werte des Trainingssets.
    X_test : pd.DataFrame oder np.ndarray, Form (n_test_samples, n_features)
        Testmerkmale (Features), dieselbe Spaltenstruktur wie X_train.
    y_test : pd.Series oder np.ndarray, Form (n_test_samples,)
        Die tatsächlichen RUL-Werte des Testsets.
    models : list of tuples (name, estimator, supports_weights)
        - name (str): Eindeutiger Modellname (z. B. "Random Forest").
        - estimator: Eine unveränderte Instanz des sklearn-Modells
                     (z. B. RandomForestRegressor()).
        - supports_weights (bool): True, wenn für dieses Modell Sample-Gewichte
                                   beim Training berücksichtigt werden sollen.
    param_grids : dict
        Schlüssel: Modellname (str) wie in models, Wert: dict mit Hyperparameter-Gitter
        für GridSearchCV. Leeres Dict {} bedeutet keine Hyperparameter-Optimierung.

    Returns
    -------
    pd.DataFrame
        DataFrame mit folgenden Spalten, sortiert nach "NASA-Score" aufsteigend (kleinerer Score = besser):
            - "Model": Modellname (str).
            - "Best Params": Dictionary der besten Hyperparameter für dieses Modell.
            - "RMSE-Test": Root Mean Squared Error auf den Testdaten (float).
            - "R²-Test": Determinationskoeffizient R² auf den Testdaten (float).
            - "NASA-Score": NASA-Score auf den Testdaten (float).
    """
    results = []

    for name, model, supports_weights in models:
        params = param_grids.get(name, {})
        grid = GridSearchCV(estimator=model, param_grid=params, scoring=nasa_scorer, cv=3, n_jobs=-1)

        if supports_weights:
            # Beispiel für Gewichtung: Je höher RUL, desto kleiner das Gewicht
            weights = 1 + 2 * np.exp(-y_train / 25)
            grid.fit(X_train, y_train, sample_weight=weights)
        else:
            grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        nasa = nasa_score(y_test, y_pred)

        results.append((name, grid.best_params_, rmse, r2, nasa))

    df = pd.DataFrame(results, columns=["Model", "Best Params", "RMSE-Test", "R²-Test", "NASA-Score"])
    return df.sort_values("NASA-Score")


def run_random_search(X_train, y_train, X_test, y_test, groups, models_with_dists, n_iter):
    """
    Führt RandomizedSearchCV mit GroupKFold (Gruppen-Cross-Validation) durch und bewertet
    die besten gefundenen Hyperparameter auf den Testdaten. Dabei wird für jedes Modell eine
    bestimmte Anzahl ("n_iter") Parameterkombinationen zufällig getestet.

    Parameters
    ----------
    X_train : pd.DataFrame oder np.ndarray, Form (n_train_samples, n_features)
        Trainingsmerkmale, ohne die Spalte 'unit'.
    y_train : pd.Series oder np.ndarray, Form (n_train_samples,)
        Tatsächliche RUL-Werte des Trainingssets.
    X_test : pd.DataFrame oder np.ndarray, Form (n_test_samples, n_features)
        Testmerkmale, ohne die Spalte 'unit'.
    y_test : pd.Series oder np.ndarray, Form (n_test_samples,)
        Tatsächliche RUL-Werte des Testsets.
    groups : array-like, Form (n_train_samples,)
        Gruppenzugehörigkeit für GroupKFold (z. B. eindeutige Unit-IDs), um Datenlecks zu vermeiden.
    models_with_dists : dict
        Schlüssel: Modellname (str),
        Wert: Tuple (estimator, param_distributions), wobei param_distributions
        entweder scipy.stats-Verteilungen oder Listen von Werten sind.
    n_iter : int
        Anzahl der zu testenden Parameterkombinationen pro Modell.

    Returns
    -------
    pd.DataFrame
        DataFrame mit folgenden Spalten, sortiert nach "NASA-Score" aufsteigend:
            - "Model": Modellname (str).
            - "Best Params": Dictionary der besten Hyperparameter.
            - "RMSE-Test": Root Mean Squared Error auf den Testdaten (float).
            - "R²-Test": Determinationskoeffizient R² auf den Testdaten (float).
            - "NASA-Score": NASA-Score auf den Testdaten (float).
    """
    gkf = GroupKFold(n_splits=3)
    results = []

    for name, (model, param_dist) in models_with_dists.items():
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=nasa_scorer,
            cv=gkf.split(X_train, y_train, groups=groups),
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        # Gleiche Gewichtung wie bei run_grid_search
        weights = 1 + 2 * np.exp(-y_train / 25)
        search.fit(X_train, y_train, sample_weight=weights)

        best = search.best_estimator_
        y_pred = best.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        nasa = nasa_score(y_test, y_pred)

        results.append((name, search.best_params_, rmse, r2, nasa))

    df = pd.DataFrame(results, columns=["Model", "Best Params", "RMSE-Test", "R²-Test", "NASA-Score"])
    return df.sort_values("NASA-Score")


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
