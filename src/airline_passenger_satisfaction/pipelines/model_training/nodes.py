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
    ebm_model = ExplainableBoostingClassifier(random_state=random_state)
    ebm_model.fit(train_dataset, train_label)
    return ebm_model


def calibrate_model(
    calibration_data: pd.DataFrame,
    calibration_label: pd.Series,
    trained_model: ExplainableBoostingClassifier,
) -> CalibratedClassifierCV:
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
