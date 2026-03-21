from kedro.pipeline import Node, Pipeline

from .nodes import corr_matrix_report, plot_distributions, evaluate_model_cv


def create_pipeline(**kwargs) -> Pipeline:
    """This is a simple pipeline which generates a pair of plots"""
    return Pipeline(
        [
           Node(
               func=corr_matrix_report,
               inputs="ortodoncja_corr_cols_cleanup",
               outputs="ortodoncja_corr_matrix",
               name="correlation_matrix_node"
           ),
            Node(
                func=plot_distributions,
                inputs="ortodoncja_corr_cols_cleanup",
                outputs="ortodoncja_distributions",
                name="distributions_node"
            ),
            Node(
                func=evaluate_model_cv,
                inputs=["logistic_regression_model", "X_train", "y_train", "params:logistic_regression_params",
                                                                                   "params:experiment_params"],
                outputs="logistic_regression_cv_report",
                name="logistic_regression_cv_report_node"
            ),
            Node(
                func=evaluate_model_cv,
                inputs=["SVC_model",  "X_train", "y_train", "params:svc_params", "params:experiment_params"],
                outputs="svc_cv_report",
                name="svc_cv_report_node"
            ),
            Node(
                func=evaluate_model_cv,
                inputs=["XGBoost_model",  "X_train", "y_train", "params:xgboost_params","params:experiment_params"],
                outputs="xgboost_cv_report",
                name="xgboost_cv_report_node"
            )
        ]
    )
