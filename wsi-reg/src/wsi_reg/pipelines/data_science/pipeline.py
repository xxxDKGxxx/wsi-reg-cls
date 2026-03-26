from kedro.pipeline import Node, Pipeline

from .nodes import split, train_and_tune_pipeline, evaluate_model, train_minimal_pipeline, train_categories_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=split,
                inputs=["domy", "params:split_options"],
                outputs=["X_train_raw", "X_test_raw", "y_train_raw", "y_test_raw"],
                name="split_node",
            ),
            # FULL PIPELINE
            Node(
                func=train_and_tune_pipeline,
                inputs=["X_train_raw", "y_train_raw", "params:param_grid_full_pipeline", "params:rkf_params"],
                outputs=["best_pipeline", "best_params", "tuning_metrics", "all_experiments_results"],
                name="train_and_tune_pipeline_node",
            ),
            Node(
                func=evaluate_model,
                inputs=["best_pipeline", "X_test_raw", "y_test_raw"],
                outputs="evaluation_metrics",
                name="evaluate_model_node",
            ),
            # ONLY MAPPINGS
            Node(
                func=train_minimal_pipeline,
                inputs=["X_train_raw", "y_train_raw", "params:param_grid_minimal", "params:rkf_params"],
                outputs=["best_pipeline_minimal", "best_params_minimal", "tuning_metrics_minimal",
                         "all_experiments_results_minimal"],
                name="train_minimal_pipeline_node",
            ),
            Node(
                func=evaluate_model,
                inputs=["best_pipeline_minimal", "X_test_raw", "y_test_raw"],
                outputs="evaluation_metrics_minimal",
                name="evaluate_minimal_node",
            ),
            # no numerical cleanup
            Node(
                func=train_categories_pipeline,
                inputs=["X_train_raw", "y_train_raw", "params:param_grid_categories", "params:rkf_params"],
                outputs=["best_pipeline_categories", "best_params_categories", "tuning_metrics_categories",
                         "all_experiments_results_categories"],
                name="train_categories_pipeline_node",
            ),
            Node(
                func=evaluate_model,
                inputs=["best_pipeline_categories", "X_test_raw", "y_test_raw"],
                outputs="evaluation_metrics_categories",
                name="evaluate_categories_node",
            ),
        ]
    )
