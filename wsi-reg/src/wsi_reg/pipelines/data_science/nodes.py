from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, cross_validate
from sklearn.metrics import root_mean_squared_error, r2_score
from category_encoders import TargetEncoder
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from .transformers import (
    DataFrameMissingValuesImputer,
    DataFrameNumericalEngineer,
    DataFrameStaticMappingsEncoder,
    DataFrameDominantTextColumnDropper,
    DataFrameRareCategoryGrouper,
    DataFrameOneHotEncoder, DataFrameTargetEncoder, DataFrameCategoryConverter, DataFrameStandardScaler
)
from sklearn.model_selection import GridSearchCV

def split(df: pd.DataFrame, parameters: dict, rs_params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=parameters['test_size'],
        random_state=rs_params['random_state']
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(pipeline: Pipeline, X_test_raw: pd.DataFrame, y_test_raw: pd.Series) -> dict:
    predictions = pipeline.predict(X_test_raw)
    rmse = root_mean_squared_error(y_test_raw, predictions)
    r2 = r2_score(y_test_raw, predictions)

    rmsle = float(np.sqrt(
        np.mean((np.log1p(predictions) - np.log1p(y_test_raw.values)) ** 2)
    ))

    mape = float(np.mean(np.abs((y_test_raw.values - predictions) / y_test_raw.values)) * 100)

    metrics = {
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }

    print(f"Test RMSE:  {rmse:.2f}")
    print(f"Test MAPE:  {mape:.2f}%")
    print(f"Test R²:    {r2:.4f}")

    return metrics

def _format_cv_table(cv_results: dict) -> pd.DataFrame:
    metrics_map = {
        "rmse": -cv_results["test_rmse"],
        "r2": cv_results["test_r2"],
        "mape": -cv_results["test_mape"] * 100
    }

    df = pd.DataFrame(metrics_map).T
    df["Mean"] = df.mean(axis=1)
    df["Std"] = df.iloc[:, :-1].std(axis=1)

    df.index.name = "Metryka"
    return df.round(3).reset_index()

def train_xgboost_cv_simple(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("one_hot_encoder", DataFrameOneHotEncoder()),
        ("xgboost", xgb.XGBRegressor(objective="reg:squarederror", random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_rf_cv_simple(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("one_hot_encoder", DataFrameOneHotEncoder()),
        ("random_forest", RandomForestRegressor(random_state=rs_params["random_state"], n_jobs=-1))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_lr_cv_simple(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("scaler", DataFrameStandardScaler()),
        ("one_hot_encoder", DataFrameOneHotEncoder()),
        ("ridge_regression", Ridge(random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_xgboost_cv_categories(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("xgboost", xgb.XGBRegressor(objective="reg:squarederror", random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_rf_cv_categories(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("random_forest", RandomForestRegressor(random_state=rs_params["random_state"], n_jobs=-1))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_lr_cv_categories(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("scaler", DataFrameStandardScaler()),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("ridge_regression", Ridge(random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_xgboost_cv_numerical(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("numerical_engineer", DataFrameNumericalEngineer()),
        ("one_hot_encoder", DataFrameOneHotEncoder()),
        ("xgboost", xgb.XGBRegressor(objective="reg:squarederror", random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_rf_cv_numerical(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("numerical_engineer", DataFrameNumericalEngineer()),
        ("one_hot_encoder", DataFrameOneHotEncoder()),
        ("random_forest", RandomForestRegressor(random_state=rs_params["random_state"], n_jobs=-1))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_lr_cv_numerical(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("numerical_engineer", DataFrameNumericalEngineer()),
        ("scaler", DataFrameStandardScaler()),
        ("one_hot_encoder", DataFrameOneHotEncoder()),
        ("ridge_regression", Ridge(random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_xgboost_cv_full(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("numerical_engineer", DataFrameNumericalEngineer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("xgboost", xgb.XGBRegressor(objective="reg:squarederror", random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_rf_cv_full(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("numerical_engineer", DataFrameNumericalEngineer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("random_forest", RandomForestRegressor(random_state=rs_params["random_state"], n_jobs=-1))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_lr_cv_full(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("numerical_engineer", DataFrameNumericalEngineer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("scaler", DataFrameStandardScaler()),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("ridge_regression", Ridge(random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_xgboost_cv_target_ohe(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("xgboost", xgb.XGBRegressor(objective="reg:squarederror", random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_rf_cv_target_ohe(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("random_forest", RandomForestRegressor(random_state=rs_params["random_state"], n_jobs=-1))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_lr_cv_target_ohe(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, rs_params: dict) -> pd.DataFrame:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("scaler", DataFrameStandardScaler()),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("ridge_regression", Ridge(random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_xgboost_cv_kbest(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, param_k_best: dict, rs_params: dict) -> pd.DataFrame:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("numerical_engineer", DataFrameNumericalEngineer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("feature_selection", SelectPercentile(score_func=f_regression, percentile=param_k_best["percentile"])),
        ("xgboost", xgb.XGBRegressor(objective="reg:squarederror", random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_rf_cv_kbest(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, param_k_best: dict, rs_params: dict) -> pd.DataFrame:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("numerical_engineer", DataFrameNumericalEngineer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("feature_selection", SelectPercentile(score_func=f_regression, percentile=param_k_best["percentile"])),
        ("random_forest", RandomForestRegressor(random_state=rs_params["random_state"], n_jobs=-1))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


def train_lr_cv_kbest(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, rkf_params: dict, param_k_best: dict, rs_params: dict) -> pd.DataFrame:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("numerical_engineer", DataFrameNumericalEngineer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("scaler", DataFrameStandardScaler()),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("feature_selection", SelectPercentile(score_func=f_regression, percentile=param_k_best["percentile"])),
        ("ridge_regression", Ridge(random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train_raw,
        y=y_train_raw,
        cv=rkf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "mape": "neg_mean_absolute_percentage_error"
        },
        n_jobs=-1
    )

    return _format_cv_table(cv_results)


from sklearn.model_selection import GridSearchCV


def tune_xgboost_grid(
        X_train_raw: pd.DataFrame,
        y_train_raw: pd.Series,
        rkf_params: dict,
        param_grid: dict,
        rs_params: dict
) -> Tuple[Pipeline, Dict[str, Any]]:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("numerical_engineer", DataFrameNumericalEngineer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("xgboost", xgb.XGBRegressor(objective="reg:squarederror", random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=rkf,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1
    )

    grid_search.fit(X_train_raw, y_train_raw)

    return grid_search.best_estimator_, grid_search.best_params_


def tune_rf_grid(
        X_train_raw: pd.DataFrame,
        y_train_raw: pd.Series,
        rkf_params: dict,
        param_grid: dict,
        rs_params: dict
) -> Tuple[Pipeline, Dict[str, Any]]:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("numerical_engineer", DataFrameNumericalEngineer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("random_forest", RandomForestRegressor(random_state=rs_params["random_state"], n_jobs=-1))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=rkf,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1
    )

    grid_search.fit(X_train_raw, y_train_raw)

    return grid_search.best_estimator_, grid_search.best_params_


def tune_lr_grid(
        X_train_raw: pd.DataFrame,
        y_train_raw: pd.Series,
        rkf_params: dict,
        param_grid: dict,
        rs_params: dict
) -> Tuple[Pipeline, Dict[str, Any]]:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("numerical_engineer", DataFrameNumericalEngineer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("scaler", DataFrameStandardScaler()),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("ridge_regression", Ridge(random_state=rs_params["random_state"]))
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params["n_splits"],
        n_repeats=rkf_params["n_repeats"],
        random_state=rs_params["random_state"],
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=rkf,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1
    )

    grid_search.fit(X_train_raw, y_train_raw)

    return grid_search.best_estimator_, grid_search.best_params_