"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["train_data", "train_label", "params:random_state"],
                outputs="trained_uncalibrated_model",
                name="train_uncalibrated_model",
            ),
            # node(
            #    func=calibrate_model,
            #    inputs=["calibration_dataset", "trained_model"],
            #    outputs="calibrated_model",
            #    name="calibrate_the_model"
            # ),
            # node(
            #    func=evaluate_calibrated_model,
            #    inputs=["trained_model", "calibrated_model", "test_dataset"],
            #    outputs="score",
            #    name="evaluation"
            # )
        ]
    )
