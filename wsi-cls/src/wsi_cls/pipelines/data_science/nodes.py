import pandas as pd
from numpy.ma.core import ravel
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import compute_sample_weight

from xgboost import XGBClassifier


class CorrelatedColumnsCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.columns_to_remove_ = []

    def fit(self, X, y):
        df = X.copy()

        target_col = '__TARGET__'
        df[target_col] = y.values if isinstance(y, pd.Series) else y

        corr_matrix = df.corr(numeric_only=True)
        columns = list(X.columns)

        columns_to_remove = set()

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]

                if abs(corr_matrix.loc[col1, col2]) >= self.threshold:
                    corr1_target = abs(corr_matrix.loc[col1, target_col])
                    corr2_target = abs(corr_matrix.loc[col2, target_col])

                    if corr1_target < corr2_target:
                        columns_to_remove.add(col1)
                    else:
                        columns_to_remove.add(col2)

        self.columns_to_remove_ = list(columns_to_remove)
        return self

    def transform(self, X):
        cols_to_drop = [c for c in self.columns_to_remove_ if c in X.columns]
        return X.drop(columns=cols_to_drop)


class BalancedMultiClassXGBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=None, n_estimators=100, max_depth=3, learning_rate=0.1, **kwargs):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        self.model = XGBClassifier(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            **self.kwargs
        )

    def fit(self, X, y):
        sample_weights = compute_sample_weight(class_weight='balanced', y=y)
        self.model.fit(X, y, sample_weight=sample_weights)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        params = {
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model = XGBClassifier(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            **self.kwargs
        )
        return self

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, experiment_params: dict) -> Pipeline:

    model = make_pipeline(
        CorrelatedColumnsCleaner(threshold=experiment_params['corr_threshold']),
        SelectKBest(k=experiment_params['k_best_k']),
        StandardScaler(),
        LogisticRegression(random_state=experiment_params['random_state'], **experiment_params[
            "logistic_regression_params"]))
    model.fit(X_train, y_train)

    return model

def train_svc(X_train: pd.DataFrame, y_train: pd.Series, experiment_params: dict) -> Pipeline:
    model = make_pipeline(
        # CorrelatedColumnsCleaner(threshold=experiment_params['corr_threshold']),
        SelectKBest(k=experiment_params['k_best_k']),
        # StandardScaler(),
        SVC(random_state=experiment_params['random_state'], **experiment_params["svc_params"]))
    model.fit(X_train, y_train)

    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, experiment_params: dict) -> Pipeline:
    model = make_pipeline(
       # CorrelatedColumnsCleaner(threshold=experiment_params['corr_threshold']),
       SelectKBest(k=experiment_params['k_best_k']),
       # StandardScaler(),
        BalancedMultiClassXGBoost(random_state=experiment_params['random_state']))
    model.fit(X_train, y_train)

    return model

def grid_search_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, experiment_params: dict, grid_params: dict) -> Pipeline:
    base_model = Pipeline([
        ('correlatedcolumnscleaner', CorrelatedColumnsCleaner(threshold=experiment_params['corr_threshold'])),
        ('selectkbest', SelectKBest()),
        ('standardscaler', StandardScaler()),
        ('logisticregression', LogisticRegression(random_state=experiment_params['random_state'], **experiment_params["logistic_regression_params"]))
    ])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=experiment_params.get("random_state"))
    
    gs = GridSearchCV(base_model, grid_params, cv=cv, scoring=experiment_params['gs_scoring'], n_jobs=-1)
    gs.fit(X_train, ravel(y_train))
    print(f"Best Logistic Regression params: {gs.best_params_},  best score: {gs.best_score_}")
    return gs.best_estimator_

def grid_search_svc(X_train: pd.DataFrame, y_train: pd.Series, experiment_params: dict, grid_params: dict) -> Pipeline:
    base_model = Pipeline([
        ('selectkbest', SelectKBest()),
        ('standardscaler', StandardScaler()),
        ('svc', SVC(random_state=experiment_params['random_state'], **experiment_params["svc_params"]))
    ])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=experiment_params.get("random_state"))
    
    gs = GridSearchCV(base_model, grid_params, cv=cv, scoring=experiment_params['gs_scoring'], n_jobs=-1)
    gs.fit(X_train, ravel(y_train))
    print(f"Best SVC params: {gs.best_params_},  best score: {gs.best_score_}")
    return gs.best_estimator_

def grid_search_xgboost(X_train: pd.DataFrame, y_train: pd.Series, experiment_params: dict, grid_params: dict) -> Pipeline:
    base_model = Pipeline([
        ('selectkbest', SelectKBest()),
        ('balancedmulticlassxgboost', BalancedMultiClassXGBoost(random_state=experiment_params['random_state']))
    ])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=experiment_params.get("random_state"))

    gs = GridSearchCV(base_model, grid_params, cv=cv, scoring=experiment_params['gs_scoring'], n_jobs=-1)
    gs.fit(X_train, ravel(y_train))
    print(f"Best XGBoost params: {gs.best_params_}, best score: {gs.best_score_}")
    return gs.best_estimator_
