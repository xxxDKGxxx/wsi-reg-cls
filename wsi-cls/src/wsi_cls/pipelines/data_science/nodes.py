import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier


def split(df: pd.DataFrame, experiment_params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random_state = experiment_params['random_state']
    target_column = experiment_params['target_column_name']
    test_size = experiment_params['test_size']

    X, y = df.drop(columns=target_column), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)

    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, experiment_params: dict) -> Pipeline:
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=experiment_params['random_state']))
    model.fit(X_train, y_train)

    return model

def train_svc(X_train: pd.DataFrame, y_train: pd.Series, experiment_params: dict) -> Pipeline:
    model = make_pipeline(StandardScaler(), SVC(random_state=experiment_params['random_state']))
    model.fit(X_train, y_train)

    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, experiment_params: dict) -> Pipeline:
    model = make_pipeline(StandardScaler(), XGBClassifier(random_state=experiment_params[
        'random_state']))
    model.fit(X_train, y_train)

    return model