import pandas as pd

from .eda_sensors import (
    plot_average_sensor_trend_normalized_time,
    plot_sensor_box_violin_last_cycle,
    plot_sensor_correlation_matrix,
    plot_sensor_distributions_by_cycle_range,
    plot_sensor_rul_correlation,
    plot_single_sensor_curves,
)

# ============================================================
#             ANALYSE DER OPERATION SETTINGS
# ============================================================


def plot_opsetting_curves(df: pd.DataFrame, unit_ids, dataset_name: str = ""):
    """
    Plottet die Kurven der Operation Settings (op_setting_1 bis 3) über die Zeit
    für gegebene Units. Pro Sensor wird ein Subplot erzeugt.

    Args:
        df (pd.DataFrame): Trainingsdatensatz.
        unit_ids (int | list[int] | list[tuple[str, int]]): Unit-IDs oder (dataset, unit)-Tupel.
        dataset_name (str): Optionaler Titelzusatz.

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure.
    """
    op_cols = [f"op_setting_{i}" for i in range(1, 4)]
    fig = plot_single_sensor_curves(
        df,
        unit_ids=unit_ids,
        dataset_name=dataset_name,
        sensor_cols=op_cols,
        normalize=False,
    )
    fig.suptitle(f"Verlauf der Operation Settings {f'({dataset_name})' if dataset_name else ''}", fontsize=16)
    return fig


def plot_opsetting_correlation_matrix(df: pd.DataFrame, dataset_name: str = "", annot=True, ax=None):
    """
    Plottet die Korrelationsmatrix der Operation Settings.

    Args:
        df (pd.DataFrame): Datensatz mit op_settings.
        dataset_name (str): Optionaler Titelzusatz.
        annot (bool): Ob Korrelationswerte angezeigt werden sollen.
        ax (matplotlib.axes.Axes, optional): Falls vorhanden, wird diese Achse verwendet.

    Returns:
        matplotlib.axes.Axes: Die Achse mit der Korrelationsmatrix.
    """
    op_cols = [f"op_setting_{i}" for i in range(1, 4)]
    ax = plot_sensor_correlation_matrix(df, sensor_cols=op_cols, dataset_name=dataset_name, annot=annot, ax=ax)
    return ax


def plot_opsetting_box_violin_last_cycle(df: pd.DataFrame, dataset_name: str = ""):
    """
    Plottet Box- und Violinplots der Operation Settings für den letzten Zyklus jeder Unit.

    Args:
        df (pd.DataFrame): Datensatz mit op_settings.
        dataset_name (str): Optionaler Titelzusatz.

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure.
    """
    op_cols = [f"op_setting_{i}" for i in range(1, 4)]
    fig = plot_sensor_box_violin_last_cycle(df, sensor_cols=op_cols, dataset_name=dataset_name)
    fig.suptitle(f"Verlauf der Operation Settings {f'({dataset_name})' if dataset_name else ''}", fontsize=16)
    return fig


def plot_opsetting_distributions_by_cycle_range(df: pd.DataFrame, dataset_name: str = "", lower_quantile=0.25, upper_quantile=0.75):
    """
    Plottet die Verteilungen der Operation Settings im Vergleich kurzer vs. langer Lebensdauer.

    Args:
        df (pd.DataFrame): Datensatz.
        dataset_name (str): Optionaler Titelzusatz.
        lower_quantile (float): Schwelle für 'wenige Zyklen'.
        upper_quantile (float): Schwelle für 'viele Zyklen'.

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure.
    """
    op_cols = [f"op_setting_{i}" for i in range(1, 4)]
    fig = plot_sensor_distributions_by_cycle_range(
        df,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        dataset_name=dataset_name,
        sensor_cols=op_cols,
    )
    fig.suptitle(f"Verlauf der Operation Settings {f'({dataset_name})' if dataset_name else ''}", fontsize=16)
    return fig


def plot_average_opsetting_trend_normalized_time(df: pd.DataFrame, dataset_name: str = ""):
    """
    Plottet den durchschnittlichen Verlauf der Operation Settings über normierte Zeit
    inklusive Standardabweichung.

    Args:
        df (pd.DataFrame): Datensatz mit op_settings.
        dataset_name (str): Optionaler Titelzusatz.

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure.
    """
    op_cols = [f"op_setting_{i}" for i in range(1, 4)]
    fig = plot_average_sensor_trend_normalized_time(df, sensor_cols=op_cols, dataset_name=dataset_name)
    fig.suptitle(f"Verlauf der Operation Settings {f'({dataset_name})' if dataset_name else ''}", fontsize=16)
    return fig


def plot_opsetting_rul_correlation(df: pd.DataFrame, dataset_name: str = ""):
    """
    Plottet die Korrelation der Operation Settings mit der Lebensdauer (RUL).

    Args:
        df (pd.DataFrame): Datensatz mit op_settings.
        dataset_name (str): Optionaler Titelzusatz.

    Returns:
        matplotlib.figure.Figure: Die erzeugte Figure.
    """
    op_cols = [f"op_setting_{i}" for i in range(1, 4)]
    return plot_sensor_rul_correlation(df, sensor_cols=op_cols, dataset_name=dataset_name)
