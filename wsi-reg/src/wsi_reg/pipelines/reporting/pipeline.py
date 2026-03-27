from kedro.pipeline import Node, Pipeline

from .nodes import (
    compare_passenger_capacity_exp,
    compare_passenger_capacity_go,
    create_confusion_matrix,
)


def create_pipeline(**kwargs) -> Pipeline:
    """This is a simple pipeline which generates a pair of plots"""
    return Pipeline(
        [
        ]
    )
