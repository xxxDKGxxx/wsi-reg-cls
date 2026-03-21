import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder


def split(df: pd.DataFrame, experiment_params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random_state = experiment_params['random_state']
    target_column = experiment_params['target_column_name']
    test_size = experiment_params['test_size']

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def target_encode(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols_to_encode = ['Exterior1st', 'Exterior2nd', 'Neighborhood']

    preprocessor = ColumnTransformer(
        transformers=[
            ('target_enc', TargetEncoder(target_type='continuous'), cols_to_encode)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    X_train_encoded = preprocessor.fit_transform(X_train, y_train)
    X_test_encoded = preprocessor.transform(X_test)
    return X_train_encoded, X_test_encoded

