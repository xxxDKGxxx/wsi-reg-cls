from kedro.pipeline import Node, Pipeline

from .nodes import enc_target, increment_feature_engineering, correlated_columns_cleanup


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # Node(
            #     func=preprocess_companies,
            #     inputs="companies",
            #     outputs="preprocessed_companies",
            #     name="preprocess_companies_node",
            # ),
            # Node(
            #     func=preprocess_shuttles,
            #     inputs="shuttles",
            #     outputs="preprocessed_shuttles",
            #     name="preprocess_shuttles_node",
            # ),
            # Node(
            #     func=create_model_input_table,
            #     inputs=["preprocessed_shuttles", "preprocessed_companies", "reviews"],
            #     outputs="model_input_table",
            #     name="create_model_input_table_node",
            # ),
            Node(
                func=enc_target,
                inputs="ortodoncja",
                outputs="ortodoncja_target_enc",
                name="enc_target_node",
            ),
            Node(
                func=increment_feature_engineering,
                inputs="ortodoncja_target_enc",
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
        ]
    )
