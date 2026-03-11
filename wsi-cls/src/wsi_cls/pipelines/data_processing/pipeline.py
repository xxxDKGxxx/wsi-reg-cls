from kedro.pipeline import Node, Pipeline

from .nodes import enc_target, increment_feature_engineering, correlated_columns_cleanup


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
                func=correlated_columns_cleanup,
                inputs=["ortodoncja_increment_feat_eng",
                        "params:correlated_columns_cleanup_params"],
                outputs="ortodoncja_corr_cols_cleanup",
                name="correlated_columns_cleanup_node",
            )
            # Node(
            #     func=split,
            #     inputs=["ortodoncja_corr_cols_cleanup", "params:experiment_params"],
            #     outputs=["X_train", "X_test", "y_train", "y_test"],
            #     name="split_node"
            # ),
            # Node(
            #     func=isolation_forest_outlier_removal,
            #     inputs=["X_train", "y_train", "params:isolation_forest_params", "params:experiment_params"],
            #     outputs=["X_train_final", "y_train_final"],
            #     name="isolation_forest_outlier_removal_node",
            # )
        ]
    )
