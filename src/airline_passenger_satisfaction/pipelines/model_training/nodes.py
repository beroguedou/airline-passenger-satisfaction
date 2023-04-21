import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def train_model(train_dataset: pd.DataFrame, train_label: pd.Series, random_state: int):
    ebm_model = ExplainableBoostingClassifier(random_state=random_state)
    ebm_model.fit(train_dataset, train_label)
    return ebm_model


def calibrate_model(calibration_data, calibration_label, trained_model):
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
) -> float:
    predictions = calibrated_model.predict_proba(test_data)[:, 1]
    predictions = np.where(predictions > threshold, 1, 0)
    f1 = f1_score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    roc_auc = roc_auc_score(test_labels, predictions)
    print("yobantex =======> ", accuracy, f1, roc_auc)
    return accuracy
