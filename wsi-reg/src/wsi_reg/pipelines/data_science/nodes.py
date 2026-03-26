import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import root_mean_squared_error, r2_score
from category_encoders import TargetEncoder
import xgboost as xgb
from .transformers import (
    DataFrameMissingValuesImputer,
    DataFrameNumericalEngineer,
    DataFrameStaticMappingsEncoder,
    DataFrameDominantTextColumnDropper,
    DataFrameRareCategoryGrouper,
    DataFrameOneHotEncoder, DataFrameTargetEncoder, DataFrameCategoryConverter
)
from sklearn.model_selection import GridSearchCV

def split(df: pd.DataFrame, parameters: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=parameters['test_size'],
        random_state=parameters['random_state']
    )

    return X_train, X_test, y_train, y_test


def train_and_tune_pipeline(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, param_grid: dict, rkf_params: dict)\
        -> tuple[Pipeline, dict, dict, pd.DataFrame]:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]

    base_pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("numerical_engineer", DataFrameNumericalEngineer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("xgboost", xgb.XGBRegressor(objective="reg:squarederror")),
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params['n_splits'],
        n_repeats=rkf_params['n_repeats'],
        random_state=rkf_params['random_state']
    )

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=rkf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_raw, y_train_raw)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = float(-grid_search.best_score_)

    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df['mean_test_score'] = -cv_results_df['mean_test_score']
    cv_results_df = cv_results_df.sort_values(by='rank_test_score')

    return best_pipeline, best_params, {"best_cv_rmse": best_score}, cv_results_df


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
        "rmsle": rmsle,
        "mape": mape,
        "r2": float(r2),
    }

    print("Test RMSE:  %.2f", rmse)
    print("Test RMSLE: %.4f", rmsle)
    print("Test MAPE:  %.2f%%", mape)
    print("Test R²:    %.4f", r2)

    return metrics

def train_minimal_pipeline(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, param_grid: dict, rkf_params: dict)\
        -> tuple[Pipeline, dict, dict, pd.DataFrame]:
    base_pipeline = Pipeline(steps=[
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("category_converter", DataFrameCategoryConverter()),
        ("xgboost", xgb.XGBRegressor(
            objective="reg:squarederror",
            enable_categorical=True  # XGBoost sam obsłuży kolumny tekstowe
        )),
    ])
    rkf = RepeatedKFold(
        n_splits=rkf_params['n_splits'],
        n_repeats=rkf_params['n_repeats'],
        random_state=rkf_params['random_state']
    )

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=rkf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_raw, y_train_raw)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = float(-grid_search.best_score_)

    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df['mean_test_score'] = -cv_results_df['mean_test_score']
    cv_results_df = cv_results_df.sort_values(by='rank_test_score')
    return best_pipeline, best_params, {"best_cv_rmse": best_score}, cv_results_df


def train_categories_pipeline(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, param_grid: dict, rkf_params: dict)\
        -> tuple[Pipeline, dict, dict, pd.DataFrame]:
    cols_to_encode_target = ["Exterior1st", "Exterior2nd", "Neighborhood"]
    base_pipeline = Pipeline(steps=[
        ("missing_values", DataFrameMissingValuesImputer()),
        ("static_mappings", DataFrameStaticMappingsEncoder()),
        ("dominant_dropper", DataFrameDominantTextColumnDropper()),
        ("rare_grouper", DataFrameRareCategoryGrouper()),
        ("target_encoder", DataFrameTargetEncoder(cols=cols_to_encode_target)),
        ("one_hot_encoder", DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ("xgboost", xgb.XGBRegressor(objective="reg:squarederror")),
    ])

    rkf = RepeatedKFold(
        n_splits=rkf_params['n_splits'],
        n_repeats=rkf_params['n_repeats'],
        random_state=rkf_params['random_state']
    )

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=rkf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_raw, y_train_raw)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = float(-grid_search.best_score_)

    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df['mean_test_score'] = -cv_results_df['mean_test_score']
    cv_results_df = cv_results_df.sort_values(by='rank_test_score')
    return best_pipeline, best_params, {"best_cv_rmse": best_score}, cv_results_df