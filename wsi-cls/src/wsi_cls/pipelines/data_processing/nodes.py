import pandas as pd
from sklearn.preprocessing import LabelEncoder


#
#
# def _is_true(x: pd.Series) -> pd.Series:
#     return x == "t"
#
#
# def _parse_percentage(x: pd.Series) -> pd.Series:
#     x = x.str.replace("%", "")
#     x = x.astype(float) / 100
#     return x
#
#
# def _parse_money(x: pd.Series) -> pd.Series:
#     x = x.str.replace("$", "").str.replace(",", "")
#     x = x.astype(float)
#     return x
#
#
# def preprocess_companies(companies: pd.DataFrame) -> pd.DataFrame:
#     """Preprocesses the data for companies.
#
#     Args:
#         companies: Raw data.
#     Returns:
#         Preprocessed data, with `company_rating` converted to a float and
#         `iata_approved` converted to boolean.
#     """
#     companies["iata_approved"] = _is_true(companies["iata_approved"])
#     companies["company_rating"] = _parse_percentage(companies["company_rating"])
#     return companies
#
#
# def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
#     """Preprocesses the data for shuttles.
#
#     Args:
#         shuttles: Raw data.
#     Returns:
#         Preprocessed data, with `price` converted to a float and `d_check_complete`,
#         `moon_clearance_complete` converted to boolean.
#     """
#     shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
#     shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
#     shuttles["price"] = _parse_money(shuttles["price"])
#     return shuttles
#
#
# def create_model_input_table(
#     shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
# ) -> pd.DataFrame:
#     """Combines all data to create a model input table.
#
#     Args:
#         shuttles: Preprocessed data for shuttles.
#         companies: Preprocessed data for companies.
#         reviews: Raw data for reviews.
#     Returns:
#         Model input table.
#
#     """
#     rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
#     rated_shuttles = rated_shuttles.drop("id", axis=1)
#     model_input_table = rated_shuttles.merge(
#         companies, left_on="company_id", right_on="id"
#     )
#     model_input_table = model_input_table.dropna()
#     return model_input_table

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
#
# def isolation_forest_outlier_removal(
#         X_train: pd.DataFrame,
#         y_train: pd.DataFrame,
#         isolation_forest_params: dict,
#         experiment_params: dict) -> pd.DataFrame:
#
#     contamination = isolation_forest_params["contamination"]
#     random_state = experiment_params["random_state"]
#
#     iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
#
#     outlier_labels = iso_forest.fit_predict(df.drop(columns=['growth direction']))
#
#     df = df[outlier_labels == 1]
#
#     return df
#
# def split(df: pd.DataFrame, experiment_params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     random_state = experiment_params['random_state']
#
#     X, y = df
#     X_train, X_test, y_train, y_test = train_test_split(df, random_state=random_state)
#
#     return X_train, X_test, y_train, y_test
