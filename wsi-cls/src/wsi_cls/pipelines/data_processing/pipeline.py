from kedro.pipeline import Node, Pipeline

from .nodes import enc_target, increment_feature_engineering, split


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=enc_target,
                inputs="ortodoncja",
                outputs="ortodoncja_target_enc",
                name="enc_target_node",
            ),
            Node(
                func=increment_feature_engineering,
                inputs=["ortodoncja_target_enc", "params:feature_eng_params"],
                outputs="ortodoncja_increment_feat_eng",
                name="increment_feature_engineering_node",
            ),
            Node(
                func=split,
                inputs=["ortodoncja_increment_feat_eng", "params:experiment_params"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_node"
            )
        ]
    )
