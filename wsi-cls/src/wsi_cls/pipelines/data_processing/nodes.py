import pandas as pd
from sklearn.model_selection import train_test_split


def enc_target(df: pd.DataFrame) -> pd.DataFrame:
    df['growth direction'] = df['growth direction'].map({ 'normal': 0, 'horizontal' : 1, 'vertical': 2 })

    return df

def increment_feature_engineering(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    difference = params["difference"]
    ratio = params["ratio"]
    drop_twelve = params["drop_twelve"]

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

    if drop_twelve:
        df = df.drop(columns=twelve_y_columns)

    return df

def split(df: pd.DataFrame, experiment_params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random_state = experiment_params['random_state']
    target_column = experiment_params['target_column_name']
    test_size = experiment_params['test_size']

    X, y = df.drop(columns=target_column), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)

    return X_train, X_test, y_train, y_test