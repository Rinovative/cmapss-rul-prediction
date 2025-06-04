import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def plot_single_sensor_curves(
    df: pd.DataFrame,
    unit_ids,
    dataset_name: str = "",
    sensor_cols=None,
    normalize: bool = False,
    rolling_window: int | None = None,
):
    """
    Plottet die Sensorverläufe mehrerer Units gemeinsam (ein Subplot pro Sensor),
    optional geglättet über Rolling Mean.

    Args:
        df (pd.DataFrame): Trainingsdatensatz.
        unit_ids (int | list[int] | list[tuple[str, int]]): Unit-IDs oder (dataset, unit)-Tupel.
        dataset_name (str): Optionaler Titelzusatz.
        sensor_cols (list): Liste der Sensoren. Default: sensor_1 bis sensor_21.
        normalize (bool): Ob Sensorwerte pro Unit mit MinMax skaliert werden sollen.
        rolling_window (int | None): Fenstergrösse für Rolling Mean. None = keine Glättung.
    """
    if sensor_cols is None:
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    elif all(isinstance(i, int) for i in sensor_cols):
        sensor_cols = [f"sensor_{i}" for i in sensor_cols]

    if isinstance(unit_ids, (int, tuple)):
        unit_ids = [unit_ids]

    n = len(sensor_cols)
    ncols = 3
    nrows = -(-n // ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 3 * nrows))
    axs = axs.flatten()

    has_dataset = "dataset" in df.columns

    for uid in unit_ids:
        if has_dataset and isinstance(uid, tuple):
            dset, u = uid
            unit_df = df[(df["unit"] == u) & (df["dataset"] == dset)].copy()
            label = f"{dset} - Unit {u}"
        else:
            unit_df = df[df["unit"] == uid].copy()
            label = f"Unit {uid}"

        if normalize:
            scaler = MinMaxScaler()
            unit_df[sensor_cols] = scaler.fit_transform(unit_df[sensor_cols])

        if rolling_window is not None:
            unit_df[sensor_cols] = unit_df[sensor_cols].rolling(window=rolling_window, min_periods=1).mean()

        for i, col in enumerate(sensor_cols):
            axs[i].plot(unit_df["time"], unit_df[col], label=label, alpha=0.8)

    for i, col in enumerate(sensor_cols):
        axs[i].set_title(col)
        axs[i].set_xlabel("Zyklus")
        axs[i].set_ylabel("Wert")
        axs[i].legend(fontsize="x-small")

    for j in range(len(sensor_cols), len(axs)):
        fig.delaxes(axs[j])

    norm_text = " (normalisiert)" if normalize else ""
    smooth_text = f" – Rolling Mean (w={rolling_window})" if rolling_window else ""
    plt.suptitle(
        f"Sensorverläufe {f'({dataset_name})' if dataset_name else ''}{norm_text}{smooth_text}",
        fontsize=16,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def plot_sensor_overlay(df: pd.DataFrame, unit_id, dataset_name: str = "", sensor_cols=None, ax=None):
    """
    Normalisiert und plottet Sensorwerte einer einzelnen Unit als überlagerten Linienplot.
    Optional kann in ein bestehendes Subplot-Achsenobjekt geplottet werden.

    Args:
        df (pd.DataFrame): Datensatz mit Spalten 'unit' und Sensoren.
        unit_id (int): ID der zu visualisierenden Unit.
        dataset_name (str): Optionaler Titelzusatz (z. B. 'FD001').
        sensor_cols (list[int | str], optional): Liste der zu plottenden Sensoren.
            Kann entweder als Liste von Strings (z. B. ["sensor_2", "sensor_7"])
            oder als Liste von Zahlen (z. B. [2, 7]) übergeben werden.
            Default: alle Sensoren sensor_1 bis sensor_21.
        ax (matplotlib.axes.Axes, optional): Falls angegeben, wird in diese Achse geplottet.
            Falls None, wird eine neue Figur mit eigener Achse erstellt.

    Returns:
        matplotlib.figure.Figure | matplotlib.axes.Axes:
            Die erstellte Figure (falls ax=None) oder die übergebene Axes (falls ax übergeben wurde).
    """
    if sensor_cols is None:
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    elif all(isinstance(i, int) for i in sensor_cols):
        sensor_cols = [f"sensor_{i}" for i in sensor_cols]

    unit_df = df[df["unit"] == unit_id].copy()
    scaler = MinMaxScaler()
    unit_df[sensor_cols] = scaler.fit_transform(unit_df[sensor_cols])

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True

    for sensor in sensor_cols:
        ax.plot(unit_df["time"], unit_df[sensor], label=sensor, alpha=0.75)

    ax.set_xlabel("Zyklus")
    ax.set_ylabel("Normalisierter Sensorwert (0–1)")
    ax.set_title(f"Sensorverläufe überlagert {f'({dataset_name})' if dataset_name else ''}")

    if created_fig:
        plt.tight_layout()
        return fig
    return ax


def plot_sensor_correlation_matrix(df: pd.DataFrame, sensor_cols=None, annot=False, dataset_name="", ax=None):
    """
    Plottet die Korrelationsmatrix der Sensoren für einen gegebenen Datensatz.
    Gibt ein Figure-Objekt zurück, falls ax=None, sonst das übergebene ax.

    Args:
        df (pd.DataFrame): Datensatz mit Sensordaten.
        sensor_cols (list, optional): Liste der Sensoren, deren Korrelation angezeigt werden soll.
        annot (bool): Ob Korrelationswerte angezeigt werden sollen.
        dataset_name (str): Titelzusatz.
        ax (matplotlib.axes.Axes, optional): Optional vorhandene Achse.

    Returns:
        matplotlib.figure.Figure | matplotlib.axes.Axes: Figure wenn ax=None, sonst ax
    """
    if sensor_cols is None:
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    elif all(isinstance(i, int) for i in sensor_cols):
        sensor_cols = [f"sensor_{i}" for i in sensor_cols]

    corr = df[sensor_cols].corr()
    title = f"Korrelationsmatrix der {'Sensoren' if 'sensor' in sensor_cols[0] else 'Operation Settings'}"
    if dataset_name:
        title += f" ({dataset_name})"

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, annot=annot, fmt=".2f", ax=ax)
        ax.set_title(title)
        return fig
    else:
        sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, annot=annot, fmt=".2f", ax=ax)
        ax.set_title(title)
        return ax.figure


def plot_sensor_box_violin_last_cycle(df, sensor_cols=None, dataset_name: str = ""):
    """
    Plottet Boxplots und Violinplots für die angegebenen Sensoren basierend auf den Werten des letzten Zyklus.
    Zeigt beide Plots im gleichen Subplot nebeneinander.

    Args:
        df (pd.DataFrame): Der Datensatz mit den Sensorwerten.
        sensor_cols (list): Liste der Sensoren, die als Boxplot und Violinplot angezeigt werden sollen.
        dataset_name (str): Optionaler Titelzusatz (z. B. FD001).

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure.
    """
    if sensor_cols is None:
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    elif all(isinstance(i, int) for i in sensor_cols):
        sensor_cols = [f"sensor_{i}" for i in sensor_cols]

    last_cycle_values = df[df["time"] == df.groupby("unit")["time"].transform("max")][sensor_cols]

    n = len(sensor_cols)
    ncols = 3
    nrows = -(-n // ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows))
    axs = axs.flatten()

    for i, sensor in enumerate(sensor_cols):
        sns.violinplot(
            data=last_cycle_values[sensor],
            ax=axs[i],
            color="lightgray",
            inner=None,
            width=1.0,
        )
        sns.boxplot(data=last_cycle_values[sensor], ax=axs[i], color="lightblue", width=0.2)
        axs[i].set_title(f"{sensor} Distribution")
        axs[i].set_xlabel("Sensor")
        axs[i].set_ylabel("Wert")

    for j in range(len(sensor_cols), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.suptitle(
        f"Boxplot und Violinplot der Sensorwerte im letzten Zyklus {f'({dataset_name})' if dataset_name else ''}",
        fontsize=16,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    return fig


def plot_sensor_distributions_by_cycle_range(
    df: pd.DataFrame,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
    dataset_name: str = "",
    sensor_cols=None,
    hue_col: str = None,
):
    """
    Plottet die Verteilung der Sensordaten für alle Sensoren. Optional mit hue (z. B. Cluster).

    - Standard: Vergleich "Viele Zyklen" vs. "Wenige Zyklen"
    - Mit `hue_col`: beliebige Kategorisierung wie z. B. Clustering

    Args:
        df (pd.DataFrame): Datensatz mit Sensordaten.
        lower_quantile (float): Unteres Quantil für Lebensdauerklassifizierung.
        upper_quantile (float): Oberes Quantil für Lebensdauerklassifizierung.
        dataset_name (str): Plot-Titelzusatz.
        sensor_cols (list): Liste der zu analysierenden Sensoren.
        hue_col (str): Spaltenname zur Farbunterteilung (optional).
    """

    # Zyklenlängen und Klassengrenzen bestimmen
    cycle_lengths = df.groupby("unit")["time"].max()
    lower_cycle_limit = cycle_lengths.quantile(lower_quantile)
    upper_cycle_limit = cycle_lengths.quantile(upper_quantile)
    long_units = cycle_lengths[cycle_lengths > upper_cycle_limit].index
    short_units = cycle_lengths[cycle_lengths <= lower_cycle_limit].index
    selected_units = short_units.union(long_units)

    # Nur relevante Einträge behalten
    df = df[df["unit"].isin(selected_units)].copy()

    # Lifetime-Klassifizierung, falls kein hue
    if not hue_col:
        df["lifetime_class"] = np.where(df["unit"].isin(long_units), "Viele Zyklen", "Wenige Zyklen")

    # Sensorliste generieren
    if sensor_cols is None:
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    elif all(isinstance(i, int) for i in sensor_cols):
        sensor_cols = [f"sensor_{i}" for i in sensor_cols]

    # Plot-Grundgerüst
    n = len(sensor_cols)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 3 * nrows))
    axs = axs.flatten()

    # Einzelne Sensoren plotten
    for i, sensor in enumerate(sensor_cols):
        ax = axs[i]
        sensor_var = df[sensor].var()
        use_kde = sensor_var > 1e-4

        sns.histplot(
            data=df,
            x=sensor,
            hue=hue_col if hue_col else "lifetime_class",
            kde=use_kde,
            ax=ax,
            stat="density",
            common_norm=False,
            element="step",
            bins=50,
        )

        if ax.get_legend():
            if len(ax.get_legend().texts) <= 5:
                ax.legend_.set_title(hue_col if hue_col else "lifetime_class")
            else:
                ax.legend_.remove()

        ax.set_title(f"{sensor} Distribution")
        ax.set_xlabel("Wert")
        ax.set_ylabel("Häufigkeit")
        ax.grid(True)

    # Leere Plots ausblenden
    for j in range(len(sensor_cols), len(axs)):
        fig.delaxes(axs[j])

    # Gesamttitel und Layout
    plt.tight_layout()
    plt.suptitle(
        f"Verteilungen der Sensoren {f'({dataset_name})' if dataset_name else ''}",
        fontsize=16,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def plot_sensor_rul_correlation(df: pd.DataFrame, sensor_cols=None, dataset_name: str = ""):
    """
    Plottet die Korrelation zwischen den Sensoren und der Remaining Useful Life (RUL).

    Args:
        df (pd.DataFrame): Der Datensatz mit Sensorwerten und RUL-Spalte.
        sensor_cols (list): Liste der Sensor-Spaltennamen oder Sensor-IDs.
        dataset_name (str): Optionaler Titelzusatz.

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure.
    """

    if sensor_cols is None:
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    elif all(isinstance(i, int) for i in sensor_cols):
        sensor_cols = [f"sensor_{i}" for i in sensor_cols]

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = df[sensor_cols].corrwith(df["RUL"])

    fig, ax = plt.subplots(figsize=(10, 6))
    corr.plot(kind="bar", color="skyblue", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(
        f"Korrelation zwischen {'Sensoren' if 'sensor' in sensor_cols[0] 
                                else 'Operation Settings'} und RUL {f'({dataset_name})' if dataset_name else ''}"
    )
    ax.set_xlabel("Variablen")
    ax.set_ylabel("Korrelationskoeffizient")
    plt.tight_layout()

    return fig


def plot_average_sensor_trend_normalized_time(df: pd.DataFrame, sensor_cols=None, dataset_name: str = ""):
    """
    Plottet den durchschnittlichen Verlauf aller Sensoren über normierte Zeit,
    inklusive Standardabweichung als Schattierung.

    Args:
        df (pd.DataFrame): Trainingsdatensatz mit 'unit', 'time' und Sensorwerten.
        sensor_cols (list): Liste der Sensoren. Default: sensor_1 bis sensor_21.
        dataset_name (str): Optionaler Titelzusatz.
    """
    if sensor_cols is None:
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    elif all(isinstance(i, int) for i in sensor_cols):
        sensor_cols = [f"sensor_{i}" for i in sensor_cols]

    # MinMax-Skalierung der Sensoren
    df_scaled = df.copy()
    scaler = MinMaxScaler()
    df_scaled[sensor_cols] = scaler.fit_transform(df_scaled[sensor_cols])

    # Normierte Zeit berechnen
    df_scaled["norm_time"] = df_scaled.groupby("unit")["time"].transform(lambda x: x / x.max())

    # Zeitachsen-Gruppierung (z. B. 100 Bins)
    df_scaled["norm_bin"] = (df_scaled["norm_time"] * 100).astype(int).clip(0, 100)

    # Plot vorbereiten
    n = len(sensor_cols)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
    axs = axs.flatten()

    for i, sensor in enumerate(sensor_cols):
        grouped = df_scaled.groupby("norm_bin")[sensor]
        mean = grouped.mean()
        std = grouped.std()
        x = mean.index / 100  # zurück auf 0–1 normierte Achse

        ax = axs[i]
        ax.plot(x, mean, label="Mittelwert", color="blue")
        ax.fill_between(x, mean - std, mean + std, alpha=0.3, label="±1 Std", color="blue")
        ax.set_title(sensor)
        ax.set_xlabel("Normierte Zeit")
        ax.set_ylabel("Wert (normiert)")
        ax.grid(True)
        ax.legend(fontsize="x-small")

    # Leere Subplots entfernen
    for j in range(len(sensor_cols), len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle(
        f"Mittlerer Sensorverlauf mit Standardabweichung über normierte Zeit {f'({dataset_name})' if dataset_name else ''}",
        fontsize=16,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    return fig
