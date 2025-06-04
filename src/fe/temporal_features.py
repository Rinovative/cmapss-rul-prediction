import numpy as np
import pandas as pd
from scipy.stats import linregress


def add_rolling_and_delta_features(
    df: pd.DataFrame,
    sensor_cols: list[str] | None = None,
    windows: list[int] = [5],
    drop_time: bool = False,
) -> pd.DataFrame:
    """
    Fügt Delta-, Rolling- und Verlauf-Features für Zeitreihendaten pro Unit hinzu.

    Berechnet für jede Sensorspalte und jede Fenstergrösse:
    - Differenz zum vorherigen Wert (`diff`)
    - Rollierendes Minimum, Maximum → `range`
    - Rollierender Mittelwert (`mean`)
    - Rollierende Standardabweichung (`std`)
    - Lineare Regressionsslope & R² innerhalb des Fensters
    - Mittelwertdifferenz Anfang/Ende des Fensters

    Die Anzahl Zeilen bleibt erhalten. Alle fehlenden Werte werden vor-/rückwärts
    aufgefüllt, um Lücken am Anfang der Zeitreihen zu vermeiden.

    Parameters
    ----------
    df : pd.DataFrame
        Zeitreihendaten mit den Spalten 'unit' und 'time', sowie Sensorwerten.
    sensor_cols : list of str, optional
        Liste der Sensor-Spalten. Falls None, werden alle numerischen Spalten
        ausser 'unit', 'time' und 'RUL' verwendet.
    windows : list of int, default=[5]
        Fenstergrössen (in Zyklen), über die rollierende Merkmale berechnet werden.
    drop_time : bool, default=False
        Wenn True, wird die Spalte 'time' am Ende entfernt.

    Returns
    -------
    pd.DataFrame
        Ursprüngliches DataFrame mit zusätzlichen Feature-Spalten.
    """
    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c not in ["unit", "time", "RUL"] and np.issubdtype(df[c].dtype, np.number)]

    df = df.sort_values(["unit", "time"]).copy()
    feature_blocks = [df]

    for s in sensor_cols:
        grp = df.groupby("unit")[s]
        features = {f"{s}_diff": grp.diff()}

        for w in windows:
            roll = grp.rolling(w, min_periods=2)

            features[f"{s}_range_{w}"] = roll.max().reset_index(0, drop=True) - roll.min().reset_index(0, drop=True)
            features[f"{s}_mean_{w}"] = roll.mean().reset_index(0, drop=True)
            features[f"{s}_std_{w}"] = roll.std().reset_index(0, drop=True)

            # Slope & R²
            def _slope(arr):
                t = np.arange(len(arr))
                return linregress(t, arr).slope

            def _r2(arr):
                t = np.arange(len(arr))
                return linregress(t, arr).rvalue ** 2

            def _mean_diff(arr):
                n = len(arr)
                k = max(1, int(0.3 * n))
                return arr[-k:].mean() - arr[:k].mean()

            features[f"{s}_slope_{w}"] = roll.apply(_slope, raw=True).reset_index(0, drop=True)
            features[f"{s}_r2_{w}"] = roll.apply(_r2, raw=True).reset_index(0, drop=True)
            features[f"{s}_mean_diff_{w}"] = roll.apply(_mean_diff, raw=True).reset_index(0, drop=True)

        feature_blocks.append(pd.DataFrame(features))

    df_out = pd.concat(feature_blocks, axis=1).bfill().ffill()

    if drop_time and "time" in df_out.columns:
        df_out = df_out.drop(columns=["time"])

    return df_out


def compress_last_cycle_per_unit(
    df: pd.DataFrame, sensor_cols: list[str] | None = None, windows: list[int] = [5], target_col: str = "RUL"
) -> pd.DataFrame:
    """
    Führt add_rolling_and_delta_features + Reduktion auf letzte Zeile pro Unit aus.

    Args:
        df (pd.DataFrame): Rohdaten.
        sensor_cols (list of str, optional): Sensoren.
        windows (list of int): Fenstergrössen.
        target_col (str): Zielspalte (z. B. 'RUL').

    Returns:
        pd.DataFrame: Eine Zeile pro Unit.
    """
    df_fe = add_rolling_and_delta_features(
        df,
        sensor_cols=sensor_cols,
        windows=windows,
        drop_time=False,
    )
    return df_fe.sort_values(["unit", "time"]).groupby("unit").tail(1).reset_index(drop=True)


def extract_temporal_features(
    df: pd.DataFrame, sensor_cols: list[str] = None, time_points: list[float] = [0.25, 0.5, 0.75], window: float = 0.25
) -> pd.DataFrame:
    """
    Extrahiert zeitbasierte Merkmale für jede Unit und jeden Zeitbereich.
    Jede resultierende Zeile repräsentiert eine (unit, time_point)-Kombination.

    Args:
        df (pd.DataFrame): Zeitreihendaten mit normalisierter 'time'-Spalte.
        sensor_cols (list[str], optional): Sensorspalten (falls None → automatisch gewählt).
        time_points (list[float]): Normierte Zeitpunkte (0.0–1.0).
        window (float): Fensterbreite ± um jeden Zeitpunkt.

    Returns:
        pd.DataFrame: Ein DataFrame mit einer Zeile pro Unit und Zeitfenster.
    """

    if sensor_cols is None:
        sensor_cols = [col for col in df.columns if col not in ["unit", "time", "RUL"] and np.issubdtype(df[col].dtype, np.number)]

    rows = []

    for unit, group in df.groupby("unit"):
        group = group.reset_index(drop=True)
        t_series = group["time"]
        rul_series = group["RUL"]

        for t in time_points:
            mask = (t_series >= t - window) & (t_series <= t + window)
            if mask.sum() == 0:
                continue

            row = {"unit": unit, "time_point": t}
            row["RUL"] = rul_series[mask].iloc[-1]

            for col in sensor_cols:
                segment = group.loc[mask, col]
                if segment.dropna().empty:
                    continue

                row[f"{col}_min"] = segment.min()
                row[f"{col}_max"] = segment.max()
                row[f"{col}_mean"] = segment.mean()
                row[f"{col}_std"] = segment.std()
                row[f"{col}_range"] = segment.max() - segment.min()

                if len(segment) >= 2:
                    t_segment = t_series[mask]
                    reg = linregress(t_segment, segment)
                    row[f"{col}_slope"] = reg.slope
                    row[f"{col}_r2"] = reg.rvalue**2
                else:
                    row[f"{col}_slope"] = 0.0
                    row[f"{col}_r2"] = 0.0

                n = len(segment)
                early = segment.iloc[: int(0.3 * n)]
                late = segment.iloc[-int(0.3 * n) :]  # noqa: E203
                row[f"{col}_mean_diff"] = late.mean() - early.mean()

            rows.append(row)

    return pd.DataFrame(rows)


def extract_temporal_features_test(df: pd.DataFrame, sensor_cols: list[str] = None, time_point: float = 1.0, window: float = 1) -> pd.DataFrame:
    """
    Wrapper um extract_temporal_features für das Testset.
    Nutzt nur einen Zeitpunkt (letzten), mit kleinem Fenster,
    und ersetzt RUL mit originalen Labels.
    """
    # Letzten RUL je unit aus df holen
    rul_labels = df.groupby("unit").apply(lambda g: g.iloc[-1][["RUL"]]).reset_index()

    # Features extrahieren am letzten Zeitpunkt
    df_feats = extract_temporal_features(df, sensor_cols=sensor_cols, time_points=[time_point], window=window)

    # Interpolierte RUL entfernen
    df_feats = df_feats.drop(columns=["RUL"], errors="ignore")

    # Originale RUL ersetzen
    df_feats = df_feats.merge(rul_labels, on="unit", how="left")
    return df_feats
