# feature_selection.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def select_features(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, threshold: float = 0.005, rf_params: dict = None):
    """
    Führt Feature-Selection mit RandomForest durch und entfernt Features mit geringer Wichtigkeit.

    Args:
        X_train (pd.DataFrame): Trainingsdaten ohne Zielvariable.
        y_train (pd.Series): Zielvariable für das Training.
        X_test (pd.DataFrame): Testdaten ohne Zielvariable.
        threshold (float): Schwellwert für `feature_importances_` unterhalb dessen Spalten entfernt werden.
        rf_params (dict): Parameter für RandomForestRegressor (z.B. {'n_estimators':800, 'max_depth':6, ...}).

    Returns:
        X_train_reduced (pd.DataFrame): Trainingsdaten ohne unwichtige Features.
        X_test_reduced (pd.DataFrame): Testdaten ohne unwichtige Features.
        imp_df (pd.DataFrame): DataFrame mit Varname und Wichtigkeit, absteigend sortiert.
        drop_cols (list[str]): Liste der entfernten Spaltennamen.
    """
    if rf_params is None:
        rf_params = {
            "random_state": 42,
            "n_jobs": -1,
            "n_estimators": 800,
            "max_depth": 6,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }

    # RandomForest trainieren
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)

    # Feature-Importances auslesen und sortieren
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({"Varname": X_train.columns, "Imp": importances}).sort_values(by="Imp", ascending=False).reset_index(drop=True)

    # Spalten ermitteln, die unter dem Threshold liegen
    drop_cols = imp_df[imp_df["Imp"] < threshold]["Varname"].tolist()

    # Daten reduzieren
    X_train_reduced = X_train.drop(columns=drop_cols)
    X_test_reduced = X_test.drop(columns=drop_cols)

    return X_train_reduced, X_test_reduced, imp_df, drop_cols
