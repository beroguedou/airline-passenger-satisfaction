from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def merge_datasets(
    questions_dataset: pd.DataFrame, labels_datasets: pd.DataFrame, merged_column: str
) -> pd.DataFrame:
    merged_datasets = pd.merge(questions_dataset, labels_datasets, on=merged_column)
    return merged_datasets


def delete_columns(dataset: pd.DataFrame, columns_to_delete: List[str]) -> pd.DataFrame:
    dataset = dataset.drop(columns_to_delete, axis=1)
    return dataset


def rename_columns_in_dataframe(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    dataframe = dataframe.copy()
    columns_names_mapping = {}
    old_columns = dataframe.columns
    for col in old_columns:
        columns_names_mapping[col] = (
            col.lower().replace("/", "_").replace(" ", "_").replace("-", "_")
        )
    dataframe = dataframe.rename(columns=columns_names_mapping)
    return dataframe, columns_names_mapping


def compute_unique_values(
    dataframe: pd.DataFrame, catagerocial_values: List[str], label: str
) -> Dict:
    mapping_unique_values = {}
    # Allows label to be treated like other categorical
    catagerocial_values.append(label)
    for col in catagerocial_values:
        mapping_unique_values[col] = dataframe[col].unique().tolist()
    return mapping_unique_values


def encode_categorical_features(
    dataframe: pd.DataFrame, mapping_unique_values: Dict[str, List[str]]
):
    for feature in mapping_unique_values:
        unique_values = mapping_unique_values[feature]
        dataframe[feature] = dataframe[feature].apply(lambda x: unique_values.index(x))
    return dataframe


def split_dataset(
    dataframe: pd.DataFrame, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train, X_remain = train_test_split(
        dataframe, test_size=0.4, random_state=random_state
    )
    X_calibration, X_test = train_test_split(
        X_remain, test_size=0.33, random_state=random_state
    )
    return X_train, X_test, X_calibration


def separate_label(
    dataframe: pd.DataFrame, label: str
) -> Tuple[pd.DataFrame, pd.Series]:
    col_label = dataframe[label]
    dataframe = dataframe.drop(label, axis=1)
    return dataframe, col_label
