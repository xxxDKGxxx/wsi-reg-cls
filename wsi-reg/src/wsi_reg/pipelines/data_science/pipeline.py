from kedro.pipeline import Node, Pipeline

from .nodes import split, evaluate_model, train_xgboost_cv_simple, train_rf_cv_simple, train_lr_cv_simple, \
    train_xgboost_cv_categories, \
    train_rf_cv_categories, train_lr_cv_categories, train_lr_cv_numerical, train_rf_cv_numerical, \
    train_xgboost_cv_numerical, train_xgboost_cv_full, train_rf_cv_full, train_lr_cv_full, train_xgboost_cv_target_ohe, \
    train_rf_cv_target_ohe, train_lr_cv_target_ohe, train_xgboost_cv_kbest, train_rf_cv_kbest, train_lr_cv_kbest, \
    tune_xgboost_grid, tune_rf_grid, tune_lr_grid


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=split,
                inputs=["domy", "params:split_options", "params:rs_params"],
                outputs=["X_train_raw", "X_test_raw", "y_train_raw", "y_test_raw"],
                name="split_node",
            ),
            # Node(
            #     func=train_xgboost_cv_simple,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_xgboost_simple",
            #     name="cv_xgboost_simple_node",
            # ),
            # Node(
            #     func=train_rf_cv_simple,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_rf_simple",
            #     name="cv_rf_simple_node",
            # ),
            # Node(
            #     func=train_lr_cv_simple,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_lr_simple",
            #     name="cv_lr_simple_node",
            # ),
            # Node(
            #     func=train_xgboost_cv_categories,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_xgboost_categories",
            #     name="cv_xgboost_categories_node",
            # ),
            # Node(
            #     func=train_rf_cv_categories,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_rf_categories",
            #     name="cv_rf_categories_node",
            # ),
            # Node(
            #     func=train_lr_cv_categories,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_lr_categories",
            #     name="cv_lr_categories_node",
            # ),
            # Node(
            #     func=train_xgboost_cv_numerical,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_xgboost_numerical",
            #     name="cv_xgboost_numerical_node",
            # ),
            # Node(
            #     func=train_rf_cv_numerical,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_rf_numerical",
            #     name="cv_rf_numerical_node",
            # ),
            # Node(
            #     func=train_lr_cv_numerical,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_lr_numerical",
            #     name="cv_lr_numerical_node",
            # ),
            # Node(
            #     func=train_xgboost_cv_full,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_xgboost_full",
            #     name="cv_xgboost_full_node",
            # ),
            # Node(
            #     func=train_rf_cv_full,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_rf_full",
            #     name="cv_rf_full_node",
            # ),
            # Node(
            #     func=train_lr_cv_full,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_lr_full",
            #     name="cv_lr_full_node",
            # ),
            # Node(
            #     func=train_xgboost_cv_target_ohe,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_xgboost_target_ohe",
            #     name="cv_xgboost_target_ohe_node",
            # ),
            # Node(
            #     func=train_rf_cv_target_ohe,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_rf_target_ohe",
            #     name="cv_rf_target_ohe_node",
            # ),
            # Node(
            #     func=train_lr_cv_target_ohe,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:rs_params"],
            #     outputs="cv_results_lr_target_ohe",
            #     name="cv_lr_target_ohe_node",
            # ),
            # Node(
            #     func=train_xgboost_cv_kbest,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:param_k_best", "params:rs_params"],
            #     outputs="cv_results_xgboost_kbest",
            #     name="cv_xgboost_kbest_node",
            # ),
            # Node(
            #     func=train_rf_cv_kbest,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:param_k_best", "params:rs_params"],
            #     outputs="cv_results_rf_kbest",
            #     name="cv_rf_kbest_node",
            # ),
            # Node(
            #     func=train_lr_cv_kbest,
            #     inputs=["X_train_raw", "y_train_raw", "params:rkf_params", "params:param_k_best", "params:rs_params"],
            #     outputs="cv_results_lr_kbest",
            #     name="cv_lr_kbest_node",
            # ),
            Node(
                func=tune_xgboost_grid,
                inputs=[
                    "X_train_raw",
                    "y_train_raw",
                    "params:rkf_params",
                    "params:param_grid_xgb",
                    "params:rs_params"
                ],
                outputs=["best_grid_model_xgb", "best_xgb_params"],
                name="tune_xgboost_grid_node",
            ),
            Node(
                func=evaluate_model,
                inputs=["best_grid_model_xgb", "X_test_raw", "y_test_raw"],
                outputs="final_test_metrics_xgb",
                name="evaluate_model_node_xgb",
            ),
            Node(
                func=tune_rf_grid,
                inputs=[
                    "X_train_raw",
                    "y_train_raw",
                    "params:rkf_params",
                    "params:param_grid_rf",
                    "params:rs_params"
                ],
                outputs=["best_grid_model_rf", "best_rf_params"],
                name="tune_rf_grid_node",
            ),
            Node(
                func=evaluate_model,
                inputs=["best_grid_model_rf", "X_test_raw", "y_test_raw"],
                outputs="final_test_metrics_rf",
                name="evaluate_model_node_rf",
            ),
            Node(
                func=tune_lr_grid,
                inputs=[
                    "X_train_raw",
                    "y_train_raw",
                    "params:rkf_params",
                    "params:param_grid_lr",
                    "params:rs_params"
                ],
                outputs=["best_grid_model_lr", "best_lr_params"],
                name="tune_lr_grid_node",
            ),
            Node(
                func=evaluate_model,
                inputs=["best_grid_model_lr", "X_test_raw", "y_test_raw"],
                outputs="final_test_metrics_lr",
                name="evaluate_model_node_lr",
            )

        ]
    )
