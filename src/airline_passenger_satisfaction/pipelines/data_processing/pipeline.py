"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import get_size


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_size,
                inputs="satisfaction",
                outputs="intermediate",
                name="Delete-personnal-informations",
            )
        ]
    )
