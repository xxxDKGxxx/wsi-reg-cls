from kedro.pipeline import Node, Pipeline

from .nodes import split, preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=split,
                inputs=["domy_numerical_cleanup", "params:experiment_params"],
                outputs=["x_train_raw", "x_test_raw", "y_train", "y_test"],
                name="split_node"
            ),
            Node(
                func=preprocess_data,
                inputs=["x_train_raw", "x_test_raw", "y_train", "params:experiment_params"],
                outputs=["x_train_encoded", "x_test_encoded"],
                name="preprocess_data_node"
            ),
        ]
    )
