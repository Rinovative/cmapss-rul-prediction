import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_prediction_and_residuals(y_true, y_pred, model_name="Modell") -> None:
    """
    Zeigt Vorhersage vs. Wahrheit und Residuenplot.

    Parameters
    ----------
    y_true : array-like
        Wahre RUL-Werte.
    y_pred : array-like
        Vorhergesagte RUL-Werte.
    model_name : str
        Anzeigename für den Plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    axes[0].set_xlabel("True RUL")
    axes[0].set_ylabel("Predicted RUL")
    axes[0].set_title(f"{model_name} – Vorhersage vs. Wahrheit")
    axes[0].grid(True)

    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolor="k")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted RUL")
    axes[1].set_ylabel("Residual (True - Predicted)")
    axes[1].set_title(f"{model_name} – Residuenplot")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def _strip_parentheses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entfernt Klammeranhänge am Ende der Modellnamen.

    Args:
        df (pd.DataFrame): DataFrame mit einer 'Model'-Spalte.

    Returns:
        pd.DataFrame: Kopiertes DataFrame mit bereinigten Modellnamen.
    """
    df = df.copy()
    df["Model"] = df["Model"].astype(str).str.replace(r"\s*\(.*\)$", "", regex=True)
    return df


def _prepare_model_dataframe(df_vor: pd.DataFrame, df_nach: pd.DataFrame | None, score_col: str) -> pd.DataFrame:
    """
    Bereitet das DataFrame für den Plot vor. Kann Einzelvergleich oder Vorher-Nachher-Vergleich verarbeiten.

    Args:
        df_vor (pd.DataFrame): DataFrame mit Modellnamen und Scores (ggf. mit 'Variante').
        df_nach (pd.DataFrame | None): Optional zweites DataFrame für Vergleich (gleiche Struktur wie df_vor).
        score_col (str): Name der Score-Spalte (z.B. "NASA-Score").

    Returns:
        pd.DataFrame: Vereinheitlichtes DataFrame mit Spalten ['Model', score_col, 'Variante'].

    Raises:
        ValueError: Wenn Pflichtspalten fehlen.
    """
    df_vor = _strip_parentheses(df_vor)
    if df_nach is not None:
        df_nach = _strip_parentheses(df_nach)
        for df_, name in [(df_vor, "df_vor"), (df_nach, "df_nach")]:
            if "Model" not in df_.columns or score_col not in df_.columns:
                raise ValueError(f"{name} muss Spalten ['Model', '{score_col}'] enthalten.")
        df1 = df_vor[["Model", score_col]].copy()
        df1["Variante"] = "vor"
        df2 = df_nach[["Model", score_col]].copy()
        df2["Variante"] = "nach"
        return pd.concat([df1, df2], ignore_index=True)
    else:
        df = df_vor.copy()
        if "Variante" in df.columns:
            if "Model" not in df.columns or score_col not in df.columns:
                raise ValueError(f"Das kombinierte DataFrame muss Spalten ['Model', '{score_col}', 'Variante'] enthalten.")
            return df[["Model", score_col, "Variante"]].copy()
        else:
            if "Model" not in df.columns or score_col not in df.columns:
                raise ValueError(f"df_vor muss Spalten ['Model', '{score_col}'] enthalten.")
            df["Variante"] = "nach"
            return df[["Model", score_col, "Variante"]]


def _plot_model_bars(df: pd.DataFrame, score_col: str, title: str, clip_score: float) -> None:
    """
    Erstellt den Balkenplot für Modell-Scores mit optionalem Vorher-Nachher-Vergleich.

    Args:
        df (pd.DataFrame): DataFrame mit Spalten ['Model', score_col, 'Variante'].
        score_col (str): Name der Score-Spalte.
        title (str): Plot-Titel.
        clip_score (float): Obergrenze für die Y-Achse (grössere Scores werden beschnitten).
    """
    df["Score Clipped"] = df[score_col].clip(upper=clip_score)

    if "nach" in df["Variante"].unique():
        sorted_models = df[df["Variante"] == "nach"].sort_values(score_col)["Model"].unique()
    else:
        sorted_models = np.sort(df["Model"].unique())

    model_order = list(sorted_models)
    x = np.arange(len(model_order))
    width = 0.35
    palette = plt.get_cmap("tab10")
    color_map = {name: palette(i % 10) for i, name in enumerate(model_order)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, base in enumerate(model_order):
        base_df = df[df["Model"] == base]
        variants = base_df["Variante"].unique()
        for _, row in base_df.iterrows():
            xpos = x[i]
            if "vor" in variants and "nach" in variants:
                xpos += -width / 2 if row["Variante"] == "vor" else width / 2
            alpha = 0.4 if row["Variante"] == "vor" else 1.0
            hatch = "//" if row["Variante"] == "vor" else ""
            ax.bar(xpos, row["Score Clipped"], width=width, color=color_map[base], alpha=alpha, hatch=hatch, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=45, ha="right")
    ax.set_ylabel(f"{score_col} (niedriger ist besser)")
    ax.set_title(title)
    ax.grid(True, axis="y")

    if set(df["Variante"]) == {"vor", "nach"}:
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="black", alpha=1.0, label="nach"),
            plt.Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="black", alpha=0.4, hatch="//", label="vor"),
        ]
        ax.legend(handles=handles, loc="best")

    plt.tight_layout()
    plt.show()


def plot_model_scores(
    df_vor: pd.DataFrame, df_nach: pd.DataFrame | None = None, score_col: str = "NASA-Score", title: str = "Modellvergleich", clip_score: float = 2000
) -> None:
    """
    Wrapper-Funktion für den Vergleich von Modell-Scores in einem Balkendiagramm.

    Args:
        df_vor (pd.DataFrame): DataFrame mit Scores (ggf. mit 'Variante'-Spalte).
        df_nach (pd.DataFrame | None): Optionaler Vergleichs-DataFrame.
        score_col (str): Spaltenname für den Score.
        title (str): Titel des Plots.
        clip_score (float): Y-Achsenbegrenzung (Clipping).
    """
    df = _prepare_model_dataframe(df_vor, df_nach, score_col)
    _plot_model_bars(df, score_col, title, clip_score)
