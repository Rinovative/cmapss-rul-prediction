import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def plot_op_settings_histograms(df_train_02: pd.DataFrame, df_train_04: pd.DataFrame) -> None:
    """
    Erstellt Histogramme der drei operativen Settings (op_setting_1 bis op_setting_3)
    für die Trainingsdaten der Datensätze FD002 und FD004.

    Diese Visualisierung unterstützt die manuelle Auswahl geeigneter Bin-Grenzen für die
    Gruppierung in Betriebsklassen. Unterschiede zwischen den Datensätzen werden so sofort sichtbar.

    Args:
        df_train_02 (pd.DataFrame): Trainingsdaten von FD002.
        df_train_04 (pd.DataFrame): Trainingsdaten von FD004.
    """
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.flatten()

    sns.histplot(df_train_02["op_setting_1"], ax=ax[0])
    ax[0].set_title("df_train_02 - op_setting_1")
    sns.histplot(df_train_02["op_setting_2"], ax=ax[1])
    ax[1].set_title("df_train_02 - op_setting_2")
    sns.histplot(df_train_02["op_setting_3"], ax=ax[2])
    ax[2].set_title("df_train_02 - op_setting_3")

    sns.histplot(df_train_04["op_setting_1"], ax=ax[3])
    ax[3].set_title("df_train_04 - op_setting_1")
    sns.histplot(df_train_04["op_setting_2"], ax=ax[4])
    ax[4].set_title("df_train_04 - op_setting_2")
    sns.histplot(df_train_04["op_setting_3"], ax=ax[5])
    ax[5].set_title("df_train_04 - op_setting_3")

    plt.tight_layout()
    plt.show()


def assign_op_cond_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weist jeder Zeile eine kombinierte Betriebsbedingung (`op_cond`) zu,
    basierend auf den drei operativen Einstellungen:
    - Flughöhe (op_setting_1 oder z. B. op_setting_1_altitude)
    - Machzahl (op_setting_2 oder z. B. op_setting_2_mach)
    - Drosselklappenstellung (op_setting_3 oder z. B. op_setting_3_throttle_resolver_angle)

    Die Funktion erkennt die entsprechenden Spaltennamen automatisch anhand ihres Prefixes.

    Für jede Einstellung werden diskrete Kategorien vergeben:
    - `alt_cat`: 6 Bins für Flughöhe
    - `mach_cat`: 5 Bins für Machzahl
    - `tra_cat`: Schwellenwert bei 80 für Drosselstellung

    Daraus wird eine kombinierte Klassifikation `op_cond` erstellt: z. B. "A2M1T1".

    Args:
        df (pd.DataFrame): DataFrame mit Spalten für Betriebsbedingungen (original oder umbenannt).

    Returns:
        pd.DataFrame: Kopie des DataFrames mit zusätzlichen Spalten `alt_cat`, `mach_cat`, `tra_cat`, `op_cond`.
    """
    df = df.copy()

    # Spaltennamen über Prefix automatisch erkennen (robust gegen Umbenennungen)
    alt_col = [col for col in df.columns if col.startswith("op_setting_1")][0]
    mach_col = [col for col in df.columns if col.startswith("op_setting_2")][0]
    tra_col = [col for col in df.columns if col.startswith("op_setting_3")][0]

    # Binning
    alt_bins = [-np.inf, 5, 15, 22, 30, 40, np.inf]
    alt_labels = ["A0", "A1", "A2", "A3", "A4", "A5"]
    df["alt_cat"] = pd.cut(df[alt_col], bins=alt_bins, labels=alt_labels)

    mach_bins = [-np.inf, 0.2, 0.55, 0.63, 0.8, np.inf]
    mach_labels = ["M0", "M1", "M2", "M3", "M4"]
    df["mach_cat"] = pd.cut(df[mach_col], bins=mach_bins, labels=mach_labels)

    df["tra_cat"] = np.where(df[tra_col] < 80, "T0", "T1")

    # Kombinierte Bedingung
    df["op_cond"] = df["alt_cat"].astype(str) + df["mach_cat"].astype(str) + df["tra_cat"]

    return df


def get_op_cond_distribution_summary(df: pd.DataFrame) -> str:
    """
    Erstellt eine Übersicht über die Gruppengrössen der kombinierten Betriebszustände
    (`alt_cat`, `mach_cat`, `tra_cat`) und gibt diese als formatierten String zurück.

    Diese Funktion hilft zu prüfen, ob bestimmte op_cond-Gruppen zu klein für
    eine separate Normalisierung sind (< 500 Zeilen).

    Args:
        df (pd.DataFrame): DataFrame mit bereits zugewiesenen op_cond-Binning-Spalten.

    Returns:
        str: Formatierter Bericht über minimale Gruppengrösse, Anzahl kleiner Gruppen
             und die zehn kleinsten op_cond-Buckets.
    """
    comb_counts = df.groupby(["alt_cat", "mach_cat", "tra_cat"], observed=True).size().sort_values(ascending=True)

    summary = []
    summary.append(f"Minimale Bucket-Grösse : {comb_counts.min()}")
    summary.append("\nBuckets:")
    summary.append(comb_counts.to_string())

    return "\n".join(summary)


def standardize_by_op_cond(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Führt eine z-Standardisierung aller Sensorsignale durch – separat für jede
    op_cond-Gruppe (Betriebsbedingung).

    Ziel ist es, Niveauverschiebungen durch unterschiedliche Flugbedingungen zu eliminieren,
    ohne die Sensitivität für physikalische Veränderungen zu verlieren.

    Der StandardScaler wird **ausschliesslich auf den Trainingsdaten** pro Gruppe fit-transformiert.
    Falls eine op_cond auch im Test vorkommt, wird dort `transform()` angewendet.

    Es werden automatisch alle Spalten mit Prefix 'sensor_' verwendet.

    Args:
        df_train (pd.DataFrame): Trainingsdaten mit Sensor- und op_setting-Spalten.
        df_test (pd.DataFrame): Testdaten mit denselben Spalten.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Zwei DataFrames mit standardisierten Sensorwerten pro Gruppe.
    """
    df_train = assign_op_cond_bins(df_train)
    df_test = assign_op_cond_bins(df_test)

    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()

    exclude = {"unit", "time", "RUL", "alt_cat", "mach_cat", "tra_cat", "op_cond"}
    sensor_cols = [col for col in df_train.columns if col not in exclude and pd.api.types.is_numeric_dtype(df_train[col])]

    df_train_scaled[sensor_cols] = df_train_scaled[sensor_cols].astype("float64")
    df_test_scaled[sensor_cols] = df_test_scaled[sensor_cols].astype("float64")

    for cond in df_train["op_cond"].unique():
        idx_train = df_train["op_cond"] == cond
        idx_test = df_test["op_cond"] == cond

        scaler = StandardScaler()
        df_train_scaled.loc[idx_train, sensor_cols] = scaler.fit_transform(df_train.loc[idx_train, sensor_cols])

        if idx_test.any():
            df_test_scaled.loc[idx_test, sensor_cols] = scaler.transform(df_test.loc[idx_test, sensor_cols])

    return df_train_scaled, df_test_scaled
