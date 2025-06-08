import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import PartialDependenceDisplay


def plot_feature_importance(model, X_train):
    """
    Plottet die Top-10 Feature-Wichtigkeiten eines Modells (z. B. Random Forest).

    Verwendet Gini-Importances oder ähnliche importance_-Attribute und visualisiert
    die bedeutendsten Merkmale als horizontales Balkendiagramm.

    Parameter
    ---------
    model : sklearn.BaseEstimator
        Trainiertes Modell mit Attribut `feature_importances_`.
    """
    feature_cols = [col for col in X_train.columns if col != "unit"]
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols)
    feat_imp_sorted = feat_imp.sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=feat_imp_sorted.values, y=feat_imp_sorted.index)
    plt.title("Top 10 Feature-Wichtigkeiten")
    plt.xlabel("Bedeutung (Gini Importance)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_pdp(model, X_train, top_n=6):
    """
    Erzeugt Partial Dependence Plots (PDP) für die wichtigsten Features.

    Die wichtigsten `top_n` Merkmale basierend auf feature_importances_ werden
    automatisch gewählt und in einem Grid dargestellt.

    Parameter
    ---------
    model : sklearn.BaseEstimator
        Trainiertes Modell mit `feature_importances_`-Attribut.

    top_n : int
        Anzahl der anzuzeigenden wichtigsten Features.
    """
    importances = model.feature_importances_
    feature_cols = X_train.drop(columns=["unit"], errors="ignore").columns
    top_features = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(top_n).index.tolist()

    n_cols = 3
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 3))

    PartialDependenceDisplay.from_estimator(
        model, X_train.drop(columns=["unit"], errors="ignore"), features=top_features, grid_resolution=50, ax=axes.ravel()
    )

    for ax in axes.ravel()[: len(top_features)]:
        lines = ax.get_lines()
        if lines:
            ydata = lines[0].get_ydata()
            ymin, ymax = np.min(ydata), np.max(ydata)
            if not np.isnan(ymin) and not np.isnan(ymax):
                ax.set_ylim(ymin - 1, ymax + 1)

    for ax in axes.ravel()[len(top_features) :]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_shap_beeswarm(model, X_train, X_test):
    """
    Visualisiert die globale SHAP-Wichtigkeit aller Merkmale mit einem Beeswarm-Plot.

    Zeigt sowohl die Verteilung als auch den Einfluss der Merkmale auf die Modellvorhersage
    über alle Testbeispiele hinweg.

    Parameter
    ---------
    model : sklearn.BaseEstimator
        Beliebiges Regressionsmodell, kompatibel mit SHAP.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.plots.beeswarm(shap_values)


def plot_shap_waterfalls(model, X_train, X_test, y_test, y_pred):
    """
    Zeigt drei SHAP-Waterfallplots: über-, unter- und exakt vorhergesagte Samples.

    Nutzt die Modellmittelwertsabweichung, um typische Fehlerbeispiele
    visuell zu analysieren. Ideal zur lokalen Interpretation einzelner Vorhersagen.

    Parameter
    ---------
    model : sklearn.BaseEstimator
        Trainiertes Modell.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    expected_value = shap_values.base_values.mean()
    tolerance_pred = 5
    tolerance_exact = 2.0

    idx_below = np.where(y_pred < expected_value - tolerance_pred)[0][0]
    idx_above = np.where(y_pred > expected_value + tolerance_pred)[0][0]
    idx_exact = np.argmin(np.abs(y_pred - y_test.values))
    assert abs(y_pred[idx_exact] - y_test.values[idx_exact]) < tolerance_exact

    indices = [idx_below, idx_above, idx_exact]
    titles = ["(A) Unter dem Durchschnitt", "(B) Über dem Durchschnitt", "(C) Genau Vorhersage"]

    for idx, title in zip(indices, titles):
        true = y_test.iloc[idx]
        pred = y_pred[idx]
        print(f"{title}: Vorhergesagt = {pred:.1f}, Wahr = {true:.1f}, Δ = {pred - true:+.1f}")
        shap.plots.waterfall(shap_values[idx], max_display=10)
