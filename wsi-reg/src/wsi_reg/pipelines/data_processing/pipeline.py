from kedro.pipeline import Node, Pipeline

from .nodes import small_categories_cleanup, fill_missing_values


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
            )
        ]
    )
