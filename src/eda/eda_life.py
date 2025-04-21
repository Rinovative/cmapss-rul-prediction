import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ============================================================
#             Lebensdauer-Analyse-Funktionen
# ============================================================
def compute_life_stats(df: pd.DataFrame, print_extremes: bool = True) -> pd.Series:
    """
    Berechnet die Lebensdauer (Zyklen bis zum Ausfall) pro Unit.

    Args:
        df (pd.DataFrame): Datensatz mit Spalten 'unit', 'time', optional 'dataset'.
        print_extremes (bool): Wenn True, werden kürzeste und längste Lebensdauer ausgegeben.

    Returns:
        pd.Series: Lebensdauer pro Unit (ggf. mit MultiIndex bei kombinierten Datensätzen).
    """
    group_keys = ["dataset", "unit"] if "dataset" in df.columns else ["unit"]
    life_stats = df.groupby(group_keys)["time"].max()

    if print_extremes:
        shortest = life_stats.idxmin()
        longest = life_stats.idxmax()
        print(f"Kürzeste Lebensdauer: Unit {shortest} mit {life_stats[shortest]} Zyklen")
        print(f"Längste Lebensdauer: Unit {longest} mit {life_stats[longest]} Zyklen")

    return life_stats


def describe_life_stats(df: pd.DataFrame) -> str:
    """
    Gibt die deskriptive Statistik der Lebensdauer pro Unit als String zurück.

    Args:
        df (pd.DataFrame): Datensatz mit Spalten 'unit', 'time', optional 'dataset'.

    Returns:
        str: Text mit Extremwerten und deskriptiver Statistik.
    """
    life_stats = compute_life_stats(df, print_extremes=False)
    shortest = life_stats.idxmin()
    longest = life_stats.idxmax()
    output = (
        f"Kürzeste Lebensdauer: Unit {shortest} mit {life_stats[shortest]} Zyklen\n"
        f"Längste Lebensdauer: Unit {longest} mit {life_stats[longest]} Zyklen\n"
        f"{life_stats.describe()}"
    )
    return output


def plot_life_distribution(df: pd.DataFrame, dataset_name: str, ax=None) -> plt.Figure:
    """
    Plottet die Verteilung der Lebensdauer (Zyklen bis zum Ausfall) für alle Units in einem Datensatz.
    Funktioniert auch bei kombinierten Datensätzen mit Spalte 'dataset'.

    Args:
        df (pd.DataFrame): Datensatz mit Spalten 'unit' und 'time' (und optional 'dataset').
        dataset_name (str): Titel des Plots.
        ax (matplotlib.axes.Axes, optional): Achse, auf der geplottet werden soll. Default: None.

    Returns:
        matplotlib.figure.Figure: Die erzeugte oder übergebene Figure.
    """
    group_keys = ["dataset", "unit"] if "dataset" in df.columns else ["unit"]
    life_stats: pd.Series = df.groupby(group_keys)["time"].max()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(life_stats, bins=20, kde=True, ax=ax, stat="count")

    ax.set_title(f"Verteilung der Lebensdauer ({dataset_name})", fontsize=16)
    ax.set_xlabel("Zyklen", fontsize=12)
    ax.set_ylabel("Anzahl Triebwerke", fontsize=12)
    ax.grid(True)

    return ax.get_figure()
