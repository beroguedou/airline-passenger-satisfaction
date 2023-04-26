from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io.data_catalog import DataCatalog

from airline_passenger_satisfaction.pipelines.data_processing.nodes import (
    encode_categorical_features,
)


def get_catalog_and_params() -> Tuple[DataCatalog, Dict[str, Union[str, float, int]]]:
    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        context = session.load_context()
        catalog = context.catalog
        params = context.params
    return catalog, params


def _preprocess(
    request_dataframe: pd.DataFrame,
    mapping_unique_values: Dict[str, List[str]],
    data_columns_order: List[str],
) -> pd.DataFrame:
    request_dataframe = request_dataframe[data_columns_order]
    request_dataframe_encoded = encode_categorical_features(
        request_dataframe, mapping_unique_values
    )
    return request_dataframe_encoded


def _predict(
    request_dataframe: pd.DataFrame, model: ExplainableBoostingClassifier
) -> float:
    prediction_proba = model.predict_proba(request_dataframe)[0, 1]
    return prediction_proba


def _postprocess(
    predicted_proba: pd.DataFrame, mapping_label: List[str], threshold: float
) -> Dict[str, Union[float, str]]:
    prediction_class = 1 if predicted_proba > threshold else 0
    prediction = {
        "probability": predicted_proba,
        "class": mapping_label[prediction_class],
    }
    return prediction


def inference(
    request_dataframe: pd.DataFrame,
    mapping_unique_values: Dict[str, List[str]],
    data_columns_order: List[str],
    model: ExplainableBoostingClassifier,
    mapping_label: List[str],
    threshold: float,
) -> Dict[str, Union[str, float]]:
    request_data_processed = _preprocess(
        request_dataframe, mapping_unique_values, data_columns_order
    )
    predicted_proba = _predict(request_data_processed, model)
    prediction = _postprocess(predicted_proba, mapping_label, threshold)
    return prediction
