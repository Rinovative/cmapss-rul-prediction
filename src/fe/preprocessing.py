import numpy as np
import pandas as pd

OP_SETTINGS_RENAME_DICT = {
    "op_setting_1": "op_setting_1_altitude_ft",
    "op_setting_2": "op_setting_2_mach",
    "op_setting_3": "op_setting_3_throttle_resolver_angle",
}

SENSOR_RENAME_DICT = {
    f"sensor_{i}": f"sensor_{i}_{name}"
    for i, name in {
        1: "T2_fan_inlet_temp",
        2: "T24_LPC_outlet_temp",
        3: "T30_HPC_outlet_temp",
        4: "T50_LPT_outlet_temp",
        5: "P2_fan_inlet_pres",
        6: "P15_bypass_pres",
        7: "P30_HPC_outlet_pres",
        8: "Nf_fan_speed",
        9: "Nc_core_speed",
        10: "epr_engine_pressure_ratio",
        11: "Ps30_HPC_static_pres",
        12: "phi_fuel_flow_per_Ps30",
        13: "NRf_corrected_fan_speed",
        14: "NRc_corrected_core_speed",
        15: "BPR_bypass_ratio",
        16: "farB_fuel_air_ratio",
        17: "htBleed_bleed_enthalpy",
        18: "Nf_dmd_demanded_fan_speed",
        19: "PCNfR_dmd_demanded_corr_fan_speed",
        20: "W31_HPT_coolant_bleed",
        21: "W32_LPT_coolant_bleed",
    }.items()
}


def rename_opsettings_and_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bennenung der Sensor- und Operational-Setting-Spalten für bessere Lesbarkeit.

    Diese Funktion ersetzt die ursprünglichen Spaltennamen wie 'sensor_1' oder 'op_setting_2'
    durch aussagekräftigere Namen wie 'sensor_1_T2_fan_inlet_temp' oder
    'op_setting_2_mach'. Dadurch wird die spätere Analyse und Feature Engineering erleichtert.

    Args:
        df (pd.DataFrame): Ursprüngliches DataFrame mit Rohspaltennamen.

    Returns:
        pd.DataFrame: DataFrame mit umbenannten Spalten.
    """
    df = df.copy()
    df.rename(columns=SENSOR_RENAME_DICT, inplace=True)
    df.rename(columns=OP_SETTINGS_RENAME_DICT, inplace=True)
    return df


def truncate_train_units(df: pd.DataFrame, min_cut: int = 5, max_cut: int = 30, frac: float = 0.25, seed: int | None = 42) -> pd.DataFrame:
    """
    Kürzt zufällig eine Teilmenge der Units im Trainingsset um eine zufällige Anzahl an Zyklen am Ende.

    Parameters
    ----------
    df : pd.DataFrame
        Trainingsdaten mit 'unit' als Gruppierung.
    min_cut : int
        Minimale Anzahl an Zyklen zum Abschneiden.
    max_cut : int
        Maximale Anzahl an Zyklen zum Abschneiden.
    frac : float
        Anteil der Units, die gekürzt werden.
    seed : int | None
        Zufalls-Seed (falls None → kein gesetzter Seed).

    Returns
    -------
    pd.DataFrame
        Gekürztes DataFrame mit allen Units.
    """
    if seed is not None:
        np.random.seed(seed)

    truncated = []

    for unit, g in df.groupby("unit", sort=False):
        if np.random.rand() < frac:
            cut = np.random.randint(min_cut, max_cut + 1)
            g = g.iloc[:-cut] if len(g) > cut else g.iloc[:1]
        truncated.append(g)

    return pd.concat(truncated, ignore_index=True)


# Kombinierte Feature-Namen zur späteren Auswahl
COMBINED_FEATURE_CONFIG = {
    "FD001": [
        ("rpm_diff", lambda df: df["sensor_13_NRf_corrected_fan_speed"] - df["sensor_8_Nf_fan_speed"]),
        ("temp_to_pressure", lambda df: df["sensor_4_T50_LPT_outlet_temp"] / df["sensor_11_Ps30_HPC_static_pres"]),
        ("bleed_minus_temp", lambda df: df["sensor_17_htBleed_bleed_enthalpy"] - df["sensor_4_T50_LPT_outlet_temp"]),
        ("coolant_mean", lambda df: (df["sensor_20_W31_HPT_coolant_bleed"] + df["sensor_21_W32_LPT_coolant_bleed"]) / 2),
        ("phi_to_bpr", lambda df: df["sensor_12_phi_fuel_flow_per_Ps30"] / df["sensor_15_BPR_bypass_ratio"]),
        ("torque_ratio", lambda df: df["sensor_14_NRc_corrected_core_speed"] / df["sensor_9_Nc_core_speed"]),
        ("torque_times_bleed", lambda df: df["sensor_14_NRc_corrected_core_speed"] * df["sensor_17_htBleed_bleed_enthalpy"]),
    ],
    "FD002": [
        ("rpm_diff", lambda df: df["sensor_13_NRf_corrected_fan_speed"] - df["sensor_8_Nf_fan_speed"]),
    ],
    "FD003": [
        ("rpm_diff", lambda df: df["sensor_13_NRf_corrected_fan_speed"] - df["sensor_8_Nf_fan_speed"]),
    ],
    "FD004": [
        ("rpm_diff", lambda df: df["sensor_13_NRf_corrected_fan_speed"] - df["sensor_8_Nf_fan_speed"]),
    ],
}


def add_combined_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Fügt kombinierte Features gemäss Konfiguration für einen bestimmten Datensatz (z. B. FD001) hinzu.

    Args:
        df (pd.DataFrame): DataFrame mit umbenannten Sensoren.
        dataset_name (str): Name des Datensatzes, z. B. "FD001".

    Returns:
        pd.DataFrame: DataFrame mit neuen Features.
    """
    df = df.copy()
    for name, func in COMBINED_FEATURE_CONFIG.get(dataset_name, []):
        df[name] = func(df)
    return df


def select_columns(df: pd.DataFrame, sensor_ids: list[int], dataset_name: str, include_opsettings: bool = True) -> pd.DataFrame:
    """
    Wählt gewünschte Sensoren, kombinierte Features sowie Meta- und (optional) op_setting-Spalten aus.

    Args:
        df (pd.DataFrame): Eingabedaten.
        sensor_ids (list[int]): Liste gewünschter Sensor-IDs (z. B. [2, 3, 4]).
        dataset_name (str): C-MAPSS-Datensatzname (z. B. "FD001").
        include_opsettings (bool): Falls True, werden auch op_setting-Spalten behalten.

    Returns:
        pd.DataFrame: Gefilterter DataFrame.
    """
    selected_sensor_cols = [col for col in df.columns if any(col.startswith(f"sensor_{i}_") for i in sensor_ids)]
    op_cols = [col for col in df.columns if col.startswith("op_setting_")] if include_opsettings else []
    meta_cols = ["unit", "time", "RUL"] + [col for col in df.columns if col.startswith("dataset") or "cluster" in col]
    combined_feature_names = [name for name, _ in COMBINED_FEATURE_CONFIG.get(dataset_name, [])]

    return df[meta_cols + op_cols + selected_sensor_cols + combined_feature_names]
