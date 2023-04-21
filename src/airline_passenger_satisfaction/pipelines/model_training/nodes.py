import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier


def train_model(train_dataset: pd.DataFrame, train_label: pd.Series, random_state: int):
    ebm_model = ExplainableBoostingClassifier(random_state=random_state)
    ebm_model.fit(train_dataset, train_label)
    return ebm_model


def calibrate_model(calibration_data, trained_model):
    calibrated_model = ...
    return calibrated_model


def evaluate_calibrated_model(trained_model, calibrated_model, test_dataset):
    score = ...
    return score
