import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


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


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str, weighted=False) -> dict:
    """
    Bewertet ein Regressionsmodell für RUL mit optionaler Gewichtung der Trainingsdaten.

    Parameters
    ----------
    model : sklearn-Regressor
        Zu evaluierendes Modell.
    model_name : str
        Anzeigename des Modells.
    weighted : bool, default=False
        Ob sample_weight beim Training verwendet wird.

    Returns
    -------
    dict
        Modellname und Metriken (RMSE, R², NASA-Score).
    """
    if weighted:
        weights = 1 + 2 * np.exp(-y_train / 25)
        model.fit(X_train, y_train, sample_weight=weights)
    else:
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
