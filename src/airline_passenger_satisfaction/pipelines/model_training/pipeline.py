"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import calibrate_model, evaluate_calibrated_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["train_data", "train_label", "params:random_state"],
                outputs="trained_uncalibrated_model",
                name="train_uncalibrated_model",
            ),
            node(
                func=calibrate_model,
                inputs=[
                    "calibration_data",
                    "calibration_label",
                    "trained_uncalibrated_model",
                ],
                outputs="trained_calibrated_model",
                name="calibrate_the_model",
            ),
            node(
                func=evaluate_calibrated_model,
                inputs=[
                    "trained_calibrated_model",
                    "test_data",
                    "test_label",
                    "params:threshold",
                ],
                outputs="score",
                name="evaluation",
            ),
        ]
    )
