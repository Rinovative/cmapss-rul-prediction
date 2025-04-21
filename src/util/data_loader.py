import pandas as pd


def _add_rul_column(df: pd.DataFrame) -> pd.DataFrame:
    life_stats = df.groupby("unit")["time"].max()
    df = df.copy()
    df["RUL"] = df["unit"].map(life_stats) - df["time"]
    return df


def load_cmapss_data(dataset=None, include_test_rul=True):
    col_names = ["unit", "time", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]

    train_dfs, test_dfs = [], []
    datasets = ["FD001", "FD002", "FD003", "FD004"] if dataset is None else [dataset]
    add_dataset_col = dataset is None

    for ds in datasets:
        train_path = f"data/raw/train_{ds}.txt"
        test_path = f"data/raw/test_{ds}.txt"
        rul_path = f"data/raw/RUL_{ds}.txt"

        df_train = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
        df_test = pd.read_csv(test_path, sep=r"\s+", header=None, names=col_names)

        if add_dataset_col:
            df_train["dataset"] = ds
            df_test["dataset"] = ds

        df_train = _add_rul_column(df_train)

        if include_test_rul:
            rul_df = pd.read_csv(rul_path, header=None, names=["final_rul"])
            last_cycles = df_test.groupby("unit")["time"].max()
            rul_per_unit = pd.DataFrame(
                {
                    "unit": last_cycles.index,
                    "max_cycle": last_cycles.values,
                    "final_rul": rul_df["final_rul"].values,
                }
            )

            # Merge RUL info to full test set
            df_test = df_test.merge(rul_per_unit, on="unit", how="left")
            df_test["RUL"] = df_test["final_rul"] + (df_test["max_cycle"] - df_test["time"])
            df_test.drop(columns=["max_cycle", "final_rul"], inplace=True)

        train_dfs.append(df_train)
        test_dfs.append(df_test)

    df_train = pd.concat(train_dfs).reset_index(drop=True)
    df_test = pd.concat(test_dfs).reset_index(drop=True)

    return df_train, df_test
