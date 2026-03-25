from kedro.pipeline import Node, Pipeline

from .nodes import split, train_full_pipeline, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=split,
                inputs=["domy", "params:split_options"],
                outputs=["X_train_raw", "X_test_raw", "y_train_raw", "y_test_raw"],
                name="split_node",
            ),
            Node(
                func=train_full_pipeline,
                inputs=["X_train_raw", "y_train_raw", "params:experiment_params", "params:model_params"],
                outputs="trained_pipeline",
                name="train_full_pipeline_node",
            ),
            Node(
                func=evaluate_model,
                inputs=["trained_pipeline", "X_test_raw", "y_test_raw"],
                outputs="evaluation_metrics",
                name="evaluate_model_node",
            ),
        ]
    )
