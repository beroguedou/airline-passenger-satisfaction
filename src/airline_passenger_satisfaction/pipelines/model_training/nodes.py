import logging
from typing import Dict

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


def train_model(
    train_dataset: pd.DataFrame, train_label: pd.Series, random_state: int
) -> ExplainableBoostingClassifier:
    """
    Node to train an explanatory boosting machine model.

    Parameters
    ----------
    train_dataset: pd.DataFrame
        The training dataset.

    train_label: pd.Series
        The training label.

    random_state: int
        The random seed value.

    Returns
    -------
    ebm_model: ExplainableBoostingClassifier
        The trained model.

    """
    ebm_model = ExplainableBoostingClassifier(random_state=random_state)
    ebm_model.fit(train_dataset, train_label)
    return ebm_model


def calibrate_model(
    calibration_data: pd.DataFrame,
    calibration_label: pd.Series,
    trained_model: ExplainableBoostingClassifier,
) -> CalibratedClassifierCV:
    """
    Node that allows to calibrate a trained model.

    Parameters
    ----------
    calibration_data: pd.DataFrame
        The dataset that will be use for model calibration.

    calibration_label: pd.Series
        The label that will be use for model calibration.

    trained_model: ExplainableBoostingClassifier
        The previously trained ebm model that we want to calibrate.

    Returns
    -------
    calibrated_model: CalibratedClassifierCV
        The calibrated ebm model.
    """
    calibrated_model = CalibratedClassifierCV(
        trained_model, cv="prefit", method="isotonic"
    )
    calibrated_model.fit(calibration_data, calibration_label)
    return calibrated_model


def evaluate_calibrated_model(
    calibrated_model: CalibratedClassifierCV,
    test_data: pd.DataFrame,
    test_labels: pd.Series,
    threshold: float,
) -> Dict[str, float]:
    """
    Evaluate the calibrated model.

    Parameters
    ----------
    calibrated_model: CalibratedClassifierCV

    test_data: pd.DataFrame
        Test dataset

    test_labels: pd.Series
        Test label

    threshold: float
        The threshold that will be used to cut the probabilities outputs of the calibrated model.

    Returns
    -------
    results: Dict[str, float]
        A dictionnary that contains accuracy, f1 and roc_auc scores.

    """
    results = {}
    predictions = calibrated_model.predict_proba(test_data)[:, 1]
    predictions = np.where(predictions > threshold, 1, 0)
    f1 = f1_score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    roc_auc = roc_auc_score(test_labels, predictions)
    results["accuracy_score"] = accuracy
    results["f1_score"] = f1
    results["roc_auc_score"] = roc_auc
    logger.info("The performance of tht calibrated model are: {}".format(results))
    return results
