import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder


def enc_target(df: pd.DataFrame) -> pd.DataFrame:
    enc = LabelEncoder()
    enc.fit(df[["growth direction"]])
    df["growth direction"] = enc.transform(df[["growth direction"]])

    return df

def increment_feature_engineering(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    difference = params["difference"]
    ratio = params["ratio"]

    all_columns: list[str] = list(df.columns)

    nine_y_columns = [column for column in all_columns if column.startswith("9_")]
    twelve_y_columns = [column for column in all_columns if column.startswith("12_")]

    parameter_names = [column[2:] for column in nine_y_columns]

    if difference:
        for parameter_name in parameter_names:
            df[parameter_name + "_diff"] = df["12_" + parameter_name] - df["9_" + parameter_name]

    if ratio:
        for parameter_name in parameter_names:
            df[parameter_name + "_rat"] = df["12_" + parameter_name] / df["9_" + parameter_name]

    df = df.drop(columns=twelve_y_columns)

    return df


def correlated_columns_cleanup(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    target_column_name = params["target_column_name"]
    threshold = params["threshold"]

    corr_matrix = df.corr(numeric_only=True)

    columns = list(corr_matrix.columns)
    columns.remove(target_column_name)
    columns_len = len(columns)

    columns_to_remove = []

    for first_column_idx in range(0, columns_len):
        for second_column_idx in range(first_column_idx + 1, columns_len):
            first_column_name = columns[first_column_idx]
            second_column_name = columns[second_column_idx]

            if abs(corr_matrix.loc[first_column_name, second_column_name]) < threshold:
                continue

            column_name_to_remove = first_column_name if abs(
                corr_matrix.loc[first_column_name, target_column_name]) > abs(
                corr_matrix.loc[second_column_name, target_column_name]) else second_column_name

            columns_to_remove.append(column_name_to_remove)

    print("Removing correlated columns: ", columns_to_remove)

    df = df.drop(columns=columns_to_remove)

    return df

def select_k_best(df: pd.DataFrame, params: dict, experiment_params: dict) -> pd.DataFrame:
    k = params["k"]
    target_column_name = experiment_params["target_column_name"]

    selector = SelectKBest(k=k)
    selector.set_output(transform='pandas')

    X = selector.fit_transform(df.drop(columns=target_column_name), df[target_column_name])

    return pd.concat([X, df[target_column_name]], axis=1)