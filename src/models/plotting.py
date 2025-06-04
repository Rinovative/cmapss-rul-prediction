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


def plot_model_scores(df_vor, df_nach=None, score_col="NASA-Score", title="Modellvergleich", clip_score=2000) -> None:
    """
    Zeichnet Balken für Modell-Scores. Entweder:
      - Zwei DataFrames mit ["Model", score_col"] übergeben (df_vor und df_nach), oder
      - Ein einzelnes DataFrame (df_vor) ohne "Variante"-Spalte: dann wird jeweils ein Balken zentriert gezeichnet.

    Parameter
    ----------
    df_vor : pd.DataFrame
        Entweder "Vorher"-Ergebnisse mit ["Model", score_col"], oder das einzige DataFrame,
        wenn df_nach=None.
    df_nach : pd.DataFrame oder None
        "Nachher"-Ergebnisse mit ["Model", score_col"], oder None, wenn df_vor alle Daten enthält.
    score_col : str
        Spaltenname des Scores (default "NASA-Score").
    title : str
        Titel des Plots.
    clip_score : float
        Maximalwert für Y-Achse (Scores > clip_score werden beschnitten).
    """

    # Hilfsfunktion: Klammer-Anhänge am Ende entfernen
    def strip_parentheses(df):
        df = df.copy()
        df["Model"] = df["Model"].astype(str).str.replace(r"\s*\(.*\)$", "", regex=True)
        return df

    # 1. Vorbereitung je nachdem, ob df_nach übergeben wurde
    if df_nach is not None:
        df1 = strip_parentheses(df_vor)
        df2 = strip_parentheses(df_nach)
        for df_, name in [(df1, "df_vor"), (df2, "df_nach")]:
            if "Model" not in df_.columns or score_col not in df_.columns:
                raise ValueError(f"{name} muss Spalten ['Model', '{score_col}'] enthalten.")
        df1 = df1[["Model", score_col]].copy()
        df1["Variante"] = "vor"
        df2 = df2[["Model", score_col]].copy()
        df2["Variante"] = "nach"
        df = pd.concat([df1, df2], ignore_index=True)

    else:
        df = df_vor.copy()
        df = strip_parentheses(df)
        if "Variante" in df.columns:
            if "Model" not in df.columns or score_col not in df.columns:
                raise ValueError("Das kombinierte DataFrame muss Spalten ['Model', '{score_col}', 'Variante'] enthalten.")
            df = df[["Model", score_col, "Variante"]].copy()
        else:
            if "Model" not in df.columns or score_col not in df.columns:
                raise ValueError("df_vor muss Spalten ['Model', '{score_col}'] enthalten.")
            df = df[["Model", score_col]].copy()
            df["Variante"] = "nach"

    # 2. Clipping der Scores
    df["Score Clipped"] = df[score_col].clip(upper=clip_score)

    # 3. Sortieren: nach "nach"-Score, sonst alphabetisch
    if "nach" in df["Variante"].unique():
        sorted_models = df[df["Variante"] == "nach"].sort_values(score_col)["Model"].unique()
    else:
        sorted_models = np.sort(df["Model"].unique())

    model_order = list(sorted_models)
    x = np.arange(len(model_order))
    width = 0.35

    # 4. Farbpalette je Modell
    palette = plt.get_cmap("tab10")
    color_map = {name: palette(i % 10) for i, name in enumerate(model_order)}

    # 5. Plot erstellen
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, base in enumerate(model_order):
        base_df = df[df["Model"] == base]
        variants = base_df["Variante"].unique()

        for _, row in base_df.iterrows():
            variant = row["Variante"]
            color = color_map[base]
            has_both = ("vor" in variants) and ("nach" in variants)

            if has_both:
                xpos = x[i] - width / 2 if variant == "vor" else x[i] + width / 2
            else:
                xpos = x[i]

            alpha = 0.4 if variant == "vor" else 1.0
            hatch = "//" if variant == "vor" else ""
            ax.bar(xpos, row["Score Clipped"], width=width, color=color, alpha=alpha, hatch=hatch, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=45, ha="right")
    ax.set_ylabel(f"{score_col} (niedriger ist besser)")
    ax.set_title(title)
    ax.grid(True, axis="y")

    # 6. Legende nur, wenn beide Varianten vorhanden sind
    if set(df["Variante"]) == {"vor", "nach"}:
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="black", alpha=1.0, label="nach"),
            plt.Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="black", alpha=0.4, hatch="//", label="vor"),
        ]
        ax.legend(handles=handles, loc="best")

    plt.tight_layout()
    plt.show()
