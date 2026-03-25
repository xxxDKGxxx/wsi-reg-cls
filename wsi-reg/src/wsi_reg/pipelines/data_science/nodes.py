import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error, r2_score
from category_encoders import TargetEncoder
import xgboost as xgb
from .transformers import (
    DataFrameMissingValuesImputer,
    DataFrameNumericalEngineer,
    DataFrameStaticMappingsEncoder,
    DataFrameDominantTextColumnDropper,
    DataFrameRareCategoryGrouper,
    DataFrameOneHotEncoder
)


def split(df: pd.DataFrame, parameters: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=parameters['test_size'],
        random_state=parameters['random_state']
    )

    return X_train, X_test, y_train, y_test


def train_full_pipeline(X_train_raw: pd.DataFrame, y_train_raw: pd.Series, experiment_params: dict,
                             model_params: dict) -> Pipeline:
    cols_to_encode_target = ['Exterior1st', 'Exterior2nd', 'Neighborhood']

    target_encoder_step = ColumnTransformer(
        transformers=[
            ('target_enc', TargetEncoder(), cols_to_encode_target)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    full_pipeline = Pipeline(steps=[
        ('missing_values', DataFrameMissingValuesImputer()),
        ('numerical_engineer', DataFrameNumericalEngineer(drop_threshold=experiment_params['drop_threshold'])),
        ('static_mappings', DataFrameStaticMappingsEncoder()),
        ('dominant_dropper',
         DataFrameDominantTextColumnDropper(drop_threshold=experiment_params['drop_columns_threshold'])),
        ('rare_grouper', DataFrameRareCategoryGrouper(threshold=experiment_params['group_columns_threshold'])),
        ('target_encoder', target_encoder_step),
        ('one_hot_encoder', DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target)),
        ('xgboost', xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **model_params))
    ])

    scores = cross_val_score(full_pipeline, X_train_raw, y_train_raw, cv=5, scoring='neg_root_mean_squared_error')
    print(f"Cross-Validation RMSE: {np.mean(-scores):.2f} (+/- {np.std(-scores):.2f})")

    full_pipeline.fit(X_train_raw, y_train_raw)

    return full_pipeline


def evaluate_model(pipeline: Pipeline, X_test_raw: pd.DataFrame, y_test_raw: pd.Series) -> dict:
    predictions = pipeline.predict(X_test_raw)

    rmse = root_mean_squared_error(y_test_raw, predictions)
    r2 = r2_score(y_test_raw, predictions)

    metrics = {
        "rmse": float(rmse),
        "r2": float(r2)
    }

    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R^2: {r2:.4f}")

    return metrics