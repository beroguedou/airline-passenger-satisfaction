import logging
from typing import Dict, List, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def merge_datasets(
    questions_dataset: pd.DataFrame, labels_dataset: pd.DataFrame, merged_column: str
) -> pd.DataFrame:
    """
    Merge the features dataset and a label dataset on a given column.

    Parameters
    ----------
    questions_dataset: pd.DataFrame
        The loaded dataset which represents the features. All the answers each passengers gave to satisfaction questions, caracteristics about flights and passengers.

    labels_dataset: pd.DataFrame
        A dataframe containing the answers  "neutral or dissatisfied" or "satisfied" relatives to a flight for a given passenger.

    merged_column: str
        The name of the column on which the questions_dataset and the labels_dataset should be merged.

    Returns
    -------
    merged_datasets: pd.DataFrame
        Result of merging of questions_dataset and the labels_dataset on merged_column.

    """
    merged_datasets = pd.merge(questions_dataset, labels_dataset, on=merged_column)
    logger.info("Succesfully merged features and labels for the raw datasets.")
    return merged_datasets


def delete_columns(dataset: pd.DataFrame, columns_to_delete: List[str]) -> pd.DataFrame:
    """
    Delete given column form an input dataframe.

    Parameters
    ----------
    dataset: pd.DataFrame
        A dataframe with some columns to delete.
    columns_to_delete: List[str]
        List of columns that we want to delete

    Returns
    -------
    dataset: pd.DataFrame
        The new version of dataframe after deleting some columns.
    """

    dataset = dataset.drop(columns_to_delete, axis=1)
    return dataset


def rename_columns_in_dataframe(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Rename all the columns of a given dataframe with uncased and "_" separated word.

    Parameters
    ----------
    dataframe: pd.DataFrame
        A dataframe with some columns with bad names

    Returns
    -------
    dataframe: pd.DataFrame
        Renammed dataframe

    columns_names_mapping: Dict[str, str]
        A dictionnary with the olds columns names as keys and the new columns names as values.
    """
    dataframe = dataframe.copy()
    columns_names_mapping = {}
    old_columns = dataframe.columns
    for col in old_columns:
        new_col = col.lower().replace("/", "_").replace(" ", "_").replace("-", "_")
        if new_col == "class":
            new_col = "flight_class"
        columns_names_mapping[col] = new_col
    dataframe = dataframe.rename(columns=columns_names_mapping)
    logger.info("Succesfully renamed the features present in the dataset ...")
    return dataframe, columns_names_mapping


def compute_unique_values(
    dataframe: pd.DataFrame, categorical_values: List[str], label: str
) -> Dict[str, List[Union[str, int]]]:
    """
    Compute a list of unique values for each categorical column and label in a given dataset.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The input dataframe with categorical features

    catagorical_values: List[str]
        List of categorical features in the dataset.

    label: str
        The label of the dataset which is also categorical.

    Returns
    -------
    mapping_unique_values: Dict[str, List[Union[str, int]]]
        A dictionnary that maps a categorical column name and a list of unique values prensents in each column.
    """
    mapping_unique_values = {}
    # Allows label to be treated like other categorical
    categorical_values.append(label)
    for col in categorical_values:
        unique_values = dataframe[col].unique().tolist()
        mapping_unique_values[col] = unique_values
        logger.info(
            "The unique value list for {} is ==> {}".format(col, str(unique_values))
        )
    return mapping_unique_values


def encode_categorical_features(
    dataframe: pd.DataFrame, mapping_unique_values: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Encode categorical columns in a dataframe, that allows to make them suited for machine learning algorithm.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe.

    mapping_unique_values: Dict[str, List[str]]
        A dictionnary that maps a categorical column name and a list of unique values prensents in each column.

    Returns
    -------
    dataframe: pd.DataFrame
        Categorical encoded dataframe.
    """
    dataframe = dataframe.copy()
    for feature in mapping_unique_values:
        unique_values = mapping_unique_values[feature]
        dataframe[feature] = dataframe[feature].apply(lambda x: unique_values.index(x))
    return dataframe


def get_columns_order(dataframe: pd.DataFrame, label: str) -> List[str]:
    """
    Get the order of the columns in input dataframe after deleting the label.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe

    label: str
        Label's column name.

    Returns
    -------
    columns: A list of columns in same order that their appears in dataframe.
    """
    columns = dataframe.columns.tolist()
    columns.remove(label)
    logger.info("The columns order is: {} ".format(" => ".join(columns)))
    return columns


def split_dataset(
    dataframe: pd.DataFrame, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Node for spliting the initial dataset in train, test and calibration datasets.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe that has to be splited.

    random_sate: int
        The random seed value for splitting the datasets.

    Returns
    -------
    X_train: pd.DataFrame
        Dataframe for ml algorithm training

    X_test: pd.DataFrame
        Dataframe for ml algorithm testing

    X_calibration: pd.DataFrame
        Dataframe for ml algorithm calibration.
    """
    X_train, X_remain = train_test_split(
        dataframe, test_size=0.4, random_state=random_state
    )
    X_calibration, X_test = train_test_split(
        X_remain, test_size=0.5, random_state=random_state
    )
    logger.info(
        "They are {} samples, we will use {} for training {} for validation and {} for calibration.".format(
            len(dataframe), len(X_train), len(X_test), len(X_calibration)
        )
    )
    return X_train, X_test, X_calibration


def separate_label(
    dataframe: pd.DataFrame, label: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separated label's column from other features in the input dataframe.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe  with all the features.

    label: str
        Label's column name.

    Returns
    -------
    dataframe: pd.DataFrame
        All features present in the input dataset.

    col_label: pd.Series
        Series representing the label column.
    """
    col_label = dataframe[label]
    dataframe = dataframe.drop(label, axis=1)
    return dataframe, col_label
