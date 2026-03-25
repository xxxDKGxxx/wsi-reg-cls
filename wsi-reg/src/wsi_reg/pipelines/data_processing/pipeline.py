from kedro.pipeline import Node, Pipeline

from .nodes import small_categories_cleanup, fill_missing_values, numerical_cleanup,\
    drop_dominant_columns, group_rare_categories, encode_static_mappings


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=fill_missing_values,
                inputs="domy",
                outputs="domy_fill_missing_values",
                name="fill_missing_values_node",
            ),
            Node(
                func=small_categories_cleanup,
                inputs="domy_fill_missing_values",
                outputs="domy_small_categories_cleanup",
                name="small_categories_cleanup_node",
            ),
            Node(
                func=numerical_cleanup,
                inputs="domy_small_categories_cleanup",
                outputs="domy_numerical_cleanup",
                name="numerical_cleanup_node",
            ),
            Node(
                func=drop_dominant_columns,
                inputs=["domy_fill_missing_values", "params:experiment_params"],
                outputs="domy_drop_dominant_columns",
                name="drop_dominant_columns_node",
            ),
            Node(
                func=encode_static_mappings,
                inputs="domy_drop_dominant_columns",
                outputs="domy_encode_static_mappings",
                name="encode_static_mappings_node",
            ),
            Node(
                func=group_rare_categories,
                inputs=["domy_encode_static_mappings", "params:experiment_params"],
                outputs="domy_group_rare_categories",
                name="group_rare_categories_node",
            )
        ]
    )
