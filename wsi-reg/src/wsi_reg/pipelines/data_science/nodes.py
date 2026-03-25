import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.pipeline import make_pipeline, Pipeline
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error, r2_score

from .transformers import DataFrameMissingValuesImputer, DataFrameNumericalEngineer, \
    DataFrameStaticMappingsEncoder, DataFrameDominantTextColumnDropper, DataFrameRareCategoryGrouper, DataFrameOneHotEncoder


def split(df: pd.DataFrame, experiment_params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random_state = experiment_params['random_state']
    target_column = experiment_params['target_column_name']
    test_size = experiment_params['test_size']

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, experiment_params: dict):
    cols_to_encode_target = ['Exterior1st', 'Exterior2nd', 'Neighborhood']

    target_encoder_step = ColumnTransformer(
        transformers=[
            ('target_enc', TargetEncoder(target_type='continuous'), cols_to_encode_target)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    preprocessing_pipeline = Pipeline(steps=[
        ('missing_values', DataFrameMissingValuesImputer()),
        ('dominant_dropper', DataFrameDominantTextColumnDropper(drop_threshold=experiment_params['drop_columns_threshold'])),
        ('static_mappings', DataFrameStaticMappingsEncoder()),
        ('rare_grouper', DataFrameRareCategoryGrouper(threshold=experiment_params['group_columns_threshold'])),
        ('numerical_engineer', DataFrameNumericalEngineer(drop_threshold=experiment_params['drop_threshold'])),
        ('target_encoder', target_encoder_step),
        ('one_hot_encoder', DataFrameOneHotEncoder(exclude_cols=cols_to_encode_target))
    ])

    X_train_encoded = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_test_encoded = preprocessing_pipeline.transform(X_test)

    return X_train_encoded, X_test_encoded, preprocessing_pipeline

