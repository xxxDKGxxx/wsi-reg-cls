from kedro.pipeline import Node, Pipeline

from .nodes import corr_matrix_report, plot_distributions, evaluate_model_cv, evaluate_model_test, \
    diagnose_per_class_metrics


def create_pipeline(**kwargs) -> Pipeline:
    """This is a simple pipeline which generates a pair of plots"""
    return Pipeline(
        [
           Node(
               func=corr_matrix_report,
               inputs=["X_train", "y_train"],
               outputs="ortodoncja_corr_matrix_train",
               name="correlation_matrix_train_node"
           ),
            Node(
                func=plot_distributions,
                inputs=["X_train", "y_train"],
                outputs="ortodoncja_distributions_train",
                name="distributions_train_node"
            ),
            Node(
                func=corr_matrix_report,
                inputs=["X_test", "y_test"],
                outputs="ortodoncja_corr_matrix_test",
                name="correlation_matrix_test_node"
            ),
            Node(
                func=plot_distributions,
                inputs=["X_test", "y_test"],
                outputs="ortodoncja_distributions_test",
                name="distributions_test_node"
            ),
            Node(
                func=evaluate_model_cv,
                inputs=["logistic_regression_model",
                        "X_train",
                        "y_train",
                        "params:experiment_params"],
                outputs="logistic_regression_cv_report",
                name="logistic_regression_cv_report_node"
            ),
            Node(
                func=evaluate_model_cv,
                inputs=["SVC_model",
                        "X_train",
                        "y_train",
                        "params:experiment_params"],
                outputs="svc_cv_report",
                name="svc_cv_report_node"
            ),
            Node(
                func=evaluate_model_cv,
                inputs=["XGBoost_model",
                        "X_train",
                        "y_train",
                        "params:experiment_params"],
                outputs="xgboost_cv_report",
                name="xgboost_cv_report_node"
            ),
            Node(
                func=evaluate_model_test,
                inputs=["logistic_regression_model",
                        "X_train",
                        "y_train",
                        "X_test",
                        "y_test",
                        "params:experiment_params"],
                outputs="logistic_regression_test_report",
                name="logistic_regression_test_report_node"
            ),
            Node(
                func=evaluate_model_test,
                inputs=["SVC_model",
                        "X_train",
                        "y_train",
                        "X_test",
                        "y_test",
                        "params:experiment_params"],
                outputs="svc_test_report",
                name="svc_test_report_node"
            ),
            Node(
                func=evaluate_model_test,
                inputs=["XGBoost_model",
                        "X_train",
                        "y_train",
                        "X_test",
                        "y_test",
                        "params:experiment_params"],
                outputs="xgboost_test_report",
                name="xgboost_test_report_node"
            ),
            Node(
                func=diagnose_per_class_metrics,
                inputs=["logistic_regression_model", "X_train", "y_train"],
                outputs="logistic_regression_diagnostics",
                name="logistic_regression_diagnostics_node"
            ),
            Node(
                func=diagnose_per_class_metrics,
                inputs=["SVC_model", "X_train", "y_train"],
                outputs="svc_diagnostics",
                name="svc_diagnostics_node"
            ),
            Node(
                func=diagnose_per_class_metrics,
                inputs=["XGBoost_model", "X_train", "y_train"],
                outputs="xgboost_diagnostics",
                name="xgboost_diagnostics_node"
            )
        ]
    )
