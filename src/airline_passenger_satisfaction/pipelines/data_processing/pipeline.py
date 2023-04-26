from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    compute_unique_values,
    delete_columns,
    encode_categorical_features,
    get_columns_order,
    merge_datasets,
    rename_columns_in_dataframe,
    separate_label,
    split_dataset,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=merge_datasets,
                inputs=[
                    "satisfaction_questions",
                    "satisfaction_labels",
                    "params:merged_column",
                ],
                outputs="satisfaction",
                name="merge_data_and_labels",
            ),
            node(
                func=delete_columns,
                inputs=["satisfaction", "params:columns_to_delete"],
                outputs="intermediate",
                name="delete-unnecessary-columns",
            ),
            node(
                func=rename_columns_in_dataframe,
                inputs="intermediate",
                outputs=["primary", "names_mapping"],
                name="rename-columns",
            ),
            node(
                func=compute_unique_values,
                inputs=["primary", "params:categorical_features", "params:label"],
                outputs="mapping_unique_values",
                name="mapping-of-unique-values",
            ),
            node(
                func=encode_categorical_features,
                inputs=["primary", "mapping_unique_values"],
                outputs="encoded_features",
                name="encode-categorical-features-and-labels",
            ),
            node(
                func=get_columns_order,
                inputs=["encoded_features", "params:label"],
                outputs="data_columns_order",
                name="save-columns-order",
            ),
            node(
                func=split_dataset,
                inputs=["encoded_features", "params:random_state"],
                outputs=["train_dataset", "test_dataset", "calibration_dataset"],
                name="split-train-test-calibration",
            ),
            node(
                func=separate_label,
                inputs=["train_dataset", "params:label"],
                outputs=["train_data", "train_label"],
                name="separate-train-data-and-label",
            ),
            node(
                func=separate_label,
                inputs=["test_dataset", "params:label"],
                outputs=["test_data", "test_label"],
                name="separate-test-data-and-label",
            ),
            node(
                func=separate_label,
                inputs=["calibration_dataset", "params:label"],
                outputs=["calibration_data", "calibration_label"],
                name="separate-calibration-data-and-label",
            ),
        ]
    )
