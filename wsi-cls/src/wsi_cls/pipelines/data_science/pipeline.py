from kedro.pipeline import Node, Pipeline
from .nodes import train_logistic_regression, train_svc, train_xgboost


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=train_logistic_regression,
                inputs=["X_train", "y_train", "params:experiment_params"],
                outputs="logistic_regression_model",
                name="train_logistic_regression_model_node"
            ),
            Node(
                func=train_svc,
                inputs=["X_train", "y_train", "params:experiment_params"],
                outputs="SVC_model",
                name="train_svc_model_node"
            ),
            Node(
                func=train_xgboost,
                inputs=["X_train", "y_train", "params:experiment_params"],
                outputs="XGBoost_model",
                name="train_xgboost_model_node"
            )
        ]
    )
