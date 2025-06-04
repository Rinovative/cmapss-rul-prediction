"""
Modellevaluation und Modellverwaltung für RUL-Prediction (C-MAPSS).
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Berechnet den NASA-Scoring-Fehler für RUL-Vorhersagen.

    Bei Überschätzung (RUL zu hoch): exponentielle Strafe mit Basis 10.
    Bei Unterschätzung (RUL zu tief): exponentielle Strafe mit Basis 13.

    Parameters
    ----------
    y_true : np.ndarray
        Wahre RUL-Werte.
    y_pred : np.ndarray
        Vorhergesagte RUL-Werte.

    Returns
    -------
    float
        Gesamt-NASA-Score (je kleiner, desto besser).
    """
    delta = y_pred - y_true
    score = np.where(delta < 0, np.exp(-delta / 13) - 1, np.exp(delta / 10) - 1)
    return np.sum(score)


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model") -> dict:
    """
    Evaluierung eines Regressionsmodells ohne Gewichtung.

    Parameters
    ----------
    model : sklearn-Regressor
        Modell zur Vorhersage der RUL.
    X_train : pd.DataFrame
        Trainingsmerkmale.
    y_train : pd.Series
        Wahre RUL-Werte im Training.
    X_test : pd.DataFrame
        Testmerkmale.
    y_test : pd.Series
        Wahre RUL-Werte im Test.
    model_name : str
        Name des Modells.

    Returns
    -------
    dict
        Ergebnisse der Evaluierung (RMSE, R², NASA-Score).
    """
    model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    return {
        "Model": model_name,
        "RMSE-Train": np.sqrt(mean_squared_error(y_train, y_hat_train)),
        "R²-Train": r2_score(y_train, y_hat_train),
        "RMSE-Test": np.sqrt(mean_squared_error(y_test, y_hat_test)),
        "R²-Test": r2_score(y_test, y_hat_test),
        "NASA-Score": nasa_score(y_test, y_hat_test),
    }


def evaluate_model_weighted(model, X_train, y_train, X_test, y_test, model_name="Model") -> dict:
    """
    Evaluierung eines Regressionsmodells mit Gewichtung nach RUL.

    Niedrige RUL erhalten höhere Gewichtung (mehr Strafe für schlechte Vorhersagen
    in späten Phasen).

    Parameters
    ----------
    model : sklearn-Regressor
        Modell zur Vorhersage der RUL.
    X_train : pd.DataFrame
        Trainingsmerkmale.
    y_train : pd.Series
        Wahre RUL-Werte im Training.
    X_test : pd.DataFrame
        Testmerkmale.
    y_test : pd.Series
        Wahre RUL-Werte im Test.
    model_name : str
        Name des Modells.

    Returns
    -------
    dict
        Ergebnisse der Evaluierung (RMSE, R², NASA-Score).
    """
    weights = 1 + 2 * np.exp(-y_train / 25)
    model.fit(X_train, y_train, sample_weight=weights)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    return {
        "Model": model_name + " (weighted)",
        "RMSE-Train": np.sqrt(mean_squared_error(y_train, y_hat_train)),
        "R²-Train": r2_score(y_train, y_hat_train),
        "RMSE-Test": np.sqrt(mean_squared_error(y_test, y_hat_test)),
        "R²-Test": r2_score(y_test, y_hat_test),
        "NASA-Score": nasa_score(y_test, y_hat_test),
    }


def get_model_list(weighted: bool = False) -> list[tuple[str, object]]:
    """
    Gibt eine Liste von Modellnamen und Instanzen zurück, je nach Gewichtung.

    Parameters
    ----------
    weighted : bool, optional
        Wenn True, nur Modelle, die sample_weight unterstützen.

    Returns
    -------
    list of tuple
        Liste von (Modellname, Modellinstanz).
    """
    models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge", Ridge()),
        ("Lasso", Lasso()),
        ("ElasticNet", ElasticNet()),
        ("Decision Tree", DecisionTreeRegressor()),
        ("Random Forest", RandomForestRegressor()),
        ("Gradient Boosting", GradientBoostingRegressor()),
    ]
    if not weighted:
        models += [("KNN", KNeighborsRegressor()), ("SVR", SVR())]
    return models
