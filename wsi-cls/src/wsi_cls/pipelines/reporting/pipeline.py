from kedro.pipeline import Node, Pipeline

from .nodes import corr_matrix_report, plot_distributions


def create_pipeline(**kwargs) -> Pipeline:
    """This is a simple pipeline which generates a pair of plots"""
    return Pipeline(
        [
           Node(
               func=corr_matrix_report,
               inputs="ortodoncja_corr_cols_cleanup",
               outputs="ortodoncja_corr_matrix",
               name="correlation_matrix_node"
           ),
            Node(
                func=plot_distributions,
                inputs="ortodoncja_corr_cols_cleanup",
                outputs="ortodoncja_distributions",
                name="distributions_node"
            )
        ]
    )
