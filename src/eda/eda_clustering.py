import matplotlib.pyplot as plt
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ..util import cache_plot_pickle


def plot_tsne_dbscan_clusters(
    df: pd.DataFrame,
    dataset_name: str = "",
    feature_cols: list[str] = None,
    tsne_components: int = 2,
    dbscan_eps: float = 3.75,
    dbscan_min_samples: int = 5,
    random_state: int = 42,
    force_recompute: bool = False,
):
    """
    Führt t-SNE auf den gegebenen Sensordaten durch und wendet DBSCAN-Clustering an.
    Gibt das Matplotlib-Figure-Objekt sowie die zugehörigen Clusterlabels zurück.

    Args:
        df (pd.DataFrame): Input-DataFrame (z.B. df_train_02)
        dataset_name (str): Titelzusatz
        feature_cols (list): Liste an Sensor-/Setting-Spalten. Wenn None, wird automatisch gewählt.
        tsne_components (int): Anzahl t-SNE-Komponenten (nur 2 unterstützt)
        dbscan_eps (float): DBSCAN-Eps
        dbscan_min_samples (int): DBSCAN-Min-Samples
        random_state (int): Zufallszustand für t-SNE

    Returns:
        tuple: (matplotlib.figure.Figure, np.ndarray of cluster labels)
    """

    def _plot_tsne_dbscan_clusters_internal(df, dataset_name, tsne_components, dbscan_eps, dbscan_min_samples, random_state, feature_cols):
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col.startswith("sensor_") or col.startswith("op_setting_")]

        X = df[feature_cols]
        X_scaled = StandardScaler().fit_transform(X)

        tsne = TSNE(n_components=tsne_components, random_state=random_state)
        X_tsne = tsne.fit_transform(X_scaled)

        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels_db = dbscan.fit_predict(X_tsne)

        df_plot = pd.DataFrame(X_tsne, columns=[f"TSNE{i + 1}" for i in range(tsne_components)])
        df_plot["cluster"] = labels_db

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=df_plot,
            x="TSNE1",
            y="TSNE2",
            hue="cluster",
            palette="tab10",
            s=10,
            alpha=0.7,
            ax=ax,
        )
        ax.set_title(f"DBSCAN auf t-SNE-Reduktion ({dataset_name})")
        ax.set_xlabel("t-SNE Komponente 1")
        ax.set_ylabel("t-SNE Komponente 2")
        ax.grid(True)
        ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        return fig, labels_db

    # Anwendung des Caching-Systems
    return cache_plot_pickle(
        _plot_tsne_dbscan_clusters_internal,
        df,
        dataset_name,
        kind="figures",
        force_recompute=force_recompute,
        tsne_components=tsne_components,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        random_state=random_state,
        feature_cols=feature_cols,
    )


def plot_op_settings_vs_cluster(
    df: pd.DataFrame,
    cluster_col: str = "cluster_tsne",
    dataset_name: str = "",
) -> plt.Figure:
    """
    Plottet die Verteilung der drei op_settings je Cluster in drei übereinanderliegenden Subplots.

    Args:
        df (pd.DataFrame): DataFrame mit 'op_setting_1', 'op_setting_2', 'op_setting_3' und Clusterzuweisung.
        cluster_col (str): Name der Spalte mit Clusterlabels.
        dataset_name (str): Optionaler Titelzusatz für Plot-Titel und Caching.

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure.
    """
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    op_settings = ["op_setting_1", "op_setting_2", "op_setting_3"]

    for i, setting in enumerate(op_settings):
        sns.boxplot(data=df, x=cluster_col, y=setting, ax=axs[i])
        axs[i].set_title(setting)
        axs[i].set_ylabel("Wert")
        axs[i].grid(True)

    axs[-1].set_xlabel("Cluster")
    fig.suptitle(f"Verteilung der Operation Settings je Cluster{f' ({dataset_name})' if dataset_name else ''}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_cluster_transitions_sankey(
    df: pd.DataFrame,
    cluster_col: str = "cluster_tsne",
    num_phases: int = 10,
    dataset_name: str = "",
):
    """
    Erstellt ein Sankey-Diagramm der Cluster-Übergänge über die Lebensdauer in diskrete Phasen.

    Args:
        df (pd.DataFrame): Datensatz mit 'unit', 'time' und Cluster-Spalte.
        cluster_col (str): Spalte mit Clusterzuordnung.
        num_phases (int): Anzahl gleich grosser Zeitphasen über die Lebensdauer.
        dataset_name (str): Optionaler Titelzusatz im Plot.
    """
    df = df.copy()
    df["phase"] = df.groupby("unit")["time"].rank(pct=True)
    bins = [i / num_phases for i in range(num_phases + 1)]
    labels = [str(i + 1) for i in range(num_phases)]
    df["phase_group"] = pd.cut(df["phase"], bins=bins, labels=labels, include_lowest=True)

    cluster_phases = (
        df.groupby(["unit", "phase_group"], observed=True)[cluster_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else -1).unstack()
    )

    all_clusters = sorted(df[cluster_col].dropna().astype(int).unique())
    cluster_names = {c: f"C{c}" for c in all_clusters}

    palette = pc.qualitative.Set3
    color_map = {f"{cluster_names[c]} - Z{p}": palette[c % len(palette)] for c in all_clusters for p in range(1, num_phases + 1)}

    labels = [f"{cluster_names[c]} - Z{p}" for p in range(1, num_phases + 1) for c in all_clusters]
    label_to_index = {label: i for i, label in enumerate(labels)}

    source, target, value, link_colors = [], [], [], []

    for p in range(1, num_phases):
        from_p, to_p = str(p), str(p + 1)
        for c1 in all_clusters:
            for c2 in all_clusters:
                count = ((cluster_phases[from_p] == c1) & (cluster_phases[to_p] == c2)).sum()
                if count > 0:
                    l1, l2 = (
                        f"{cluster_names[c1]} - Z{p}",
                        f"{cluster_names[c2]} - Z{p + 1}",
                    )
                    source.append(label_to_index[l1])
                    target.append(label_to_index[l2])
                    value.append(count)
                    link_colors.append(color_map[l1])

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=[color_map[label] for label in labels],
                ),
                link=dict(source=source, target=target, value=value, color=link_colors),
            )
        ]
    )

    title = "Cluster-Übergänge über Lebensdauer"
    if dataset_name:
        title += f" ({dataset_name})"
    fig.update_layout(title_text=title, font_size=12)
    return fig


def plot_cluster_average_time(df: pd.DataFrame, cluster_col: str = "cluster_tsne", normalize_time: bool = True, dataset_name: str = "") -> plt.Figure:
    """
    Plottet den durchschnittlichen (ggf. normierten) Zykluswert, in dem jeder Cluster typischerweise auftritt.

    Args:
        df (pd.DataFrame): Datensatz mit Spalten 'unit', 'time' und Cluster-Spalte.
        cluster_col (str): Spalte mit Clusterlabels.
        normalize_time (bool): Falls True, wird Zyklus pro Unit durch max-Zyklus normiert (0–1).
        dataset_name (str): Optionaler Titelzusatz.

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure.
    """
    df = df.copy()

    if normalize_time:
        # Normiere Zeit auf [0–1] je Unit
        df["norm_time"] = df["time"] / df.groupby("unit")["time"].transform("max")
        time_col = "norm_time"
    else:
        time_col = "time"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x=cluster_col, y=time_col, ax=ax)
    ax.set_title("Verteilung des Eintrittszeitpunkts je Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Eintrittszeitpunkt (normiert)" if normalize_time else "Eintrittszeitpunkt (Zyklen)")
    ax.grid(True)
    plt.tight_layout()

    return fig


def plot_cluster_distribution_last_cycle(df: pd.DataFrame, cluster_col: str = "cluster_tsne") -> plt.Figure:
    """
    Plottet die Verteilung der Clusterzugehörigkeiten basierend auf dem letzten Zyklus jeder Unit.

    Args:
        df (pd.DataFrame): Datensatz mit den Spalten 'unit', 'time' und der Cluster-Spalte.
        cluster_col (str): Spaltenname der Cluster-Zugehörigkeit.

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure.
    """
    df_last = df.sort_values("time").groupby("unit").tail(1)

    fig, ax = plt.subplots(figsize=(8, 6))
    df_last[cluster_col].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Cluster-Verteilung (nur eindeutige Units)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Anzahl Units")
    ax.grid(True)
    plt.tight_layout()

    return fig


def plot_lifetime_boxplot_by_cluster(df: pd.DataFrame, cluster_col: str = "cluster_tsne") -> plt.Figure:
    """
    Erstellt ein Boxplot der Lebensdauer (maximale Zyklusanzahl) pro finalem Cluster.
    Für jede Unit wird der letzte bekannte Cluster betrachtet.

    Args:
        df (pd.DataFrame): Datensatz mit 'unit', 'time' und Cluster-Spalte.
        cluster_col (str): Spaltenname der Cluster-Zugehörigkeit.

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure.
    """
    # Lebensdauer pro Unit
    life_df = df.groupby("unit")["time"].max().reset_index()

    # Finaler Cluster pro Unit
    final_cluster = df.sort_values("time").groupby("unit").tail(1)[["unit", cluster_col]]

    # Mergen
    merged = life_df.merge(final_cluster, on="unit", how="left")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=merged, x=cluster_col, y="time", ax=ax)

    ax.set_title("Lebensdauer pro finalem Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Lebensdauer (Zyklen)")
    ax.grid(True)
    plt.tight_layout()

    return fig


def plot_mean_normalized_sensors_by_cluster(
    df: pd.DataFrame,
    sensor_cols: list[str] = None,
    cluster_col: str = "cluster_tsne",
    dataset_name: str = "",
) -> plt.Figure:
    """
    Plottet den Mittelwert der normalisierten Sensorwerte je Cluster.

    Args:
        df (pd.DataFrame): Datensatz mit Sensorwerten und Clusterzugehörigkeit.
        sensor_cols (list): Liste der zu analysierenden Sensoren.
        cluster_col (str): Spalte mit Clusterlabels.
        dataset_name (str): Optionaler Titelzusatz.

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure.
    """
    if sensor_cols is None:
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    elif all(isinstance(i, int) for i in sensor_cols):
        sensor_cols = [f"sensor_{i}" for i in sensor_cols]

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[sensor_cols] = scaler.fit_transform(df_scaled[sensor_cols])

    df_grouped = df_scaled.groupby(cluster_col)[sensor_cols].mean()

    fig, ax = plt.subplots(figsize=(18, 8))
    df_grouped.plot(kind="bar", ax=ax)
    ax.set_title(f"Mittlere normalisierte Sensorwerte nach Cluster {f'({dataset_name})' if dataset_name else ''}")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mittlerer normalisierter Sensorwert")
    ax.legend(title="Sensors", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)
    plt.tight_layout()

    return fig


def summarize_cluster_characteristics(
    df: pd.DataFrame,
    cluster_col: str = "cluster_tsne",
) -> pd.DataFrame:
    """
    Erstellt eine aussagekräftige Zusammenfassung pro Cluster:
    - Anzahl Datenpunkte
    - Anzahl eindeutiger Units
    - Anteil Units mit finalem Cluster = X
    - Ø Eintrittszeitpunkt (normiert)
    - Varianz Eintrittszeitpunkt (normiert)
    - Ø Lebensdauer der Units, die in diesem Cluster enden

    Args:
        df (pd.DataFrame): Datensatz mit Spalten 'unit', 'time', Clusterlabels.
        cluster_col (str): Spalte mit Clusterlabels.

    Returns:
        pd.DataFrame: Zusammenfassung pro Cluster.
    """
    df = df.copy()
    df["norm_time"] = df["time"] / df.groupby("unit")["time"].transform("max")

    group = df.groupby(cluster_col)
    summary = pd.DataFrame(index=group.size().index)

    summary["Anzahl Punkte"] = group.size()
    summary["Anzahl Units"] = group["unit"].nunique()
    summary["Ø Eintritt (normiert)"] = group["norm_time"].mean()
    summary["Varianz Eintritt"] = group["norm_time"].var()

    # Finaler Cluster je Unit
    df_last = df.sort_values("time").groupby("unit").tail(1)

    # Anteil Units mit finalem Cluster = X
    final_cluster_counts = df_last[cluster_col].value_counts(normalize=True)
    summary["Anteil finaler Units"] = summary.index.map(final_cluster_counts).fillna(0)

    # Lebensdauer pro Unit
    life_df = df.groupby("unit")["time"].max()
    df_last["life"] = df_last["unit"].map(life_df)

    # Ø Lebensdauer der Units, die in diesem Cluster enden
    avg_life_by_final_cluster = df_last.groupby(cluster_col)["life"].mean()
    summary["Ø Lebensdauer (finale Units)"] = summary.index.map(avg_life_by_final_cluster).fillna(0)

    return summary.sort_index()
