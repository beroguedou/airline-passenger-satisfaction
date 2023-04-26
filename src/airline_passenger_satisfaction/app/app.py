import pandas as pd
from flask import Flask, jsonify, request
from utils import get_catalog

from airline_passenger_satisfaction.pipelines.data_processing.nodes import (
    encode_categorical_features,
)

threshold = 0.5
catalog = get_catalog()
calibrated_model = catalog.load("trained_calibrated_model")
data_columns_order = catalog.load("data_columns_order")
mapping_unique_values = catalog.load("mapping_unique_values")
mapping_label = mapping_unique_values["satisfaction"]
del mapping_unique_values["satisfaction"]

app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello_world():
    prediction = {"probability": str(0.01), "class": "neutral"}
    return jsonify(prediction)


@app.route("/predict/satisfaction", methods=["POST"])
def predict_satisfaction():
    request_data = request.json
    request_data = {k: [v] for (k, v) in request_data.items()}
    request_dataframe = pd.DataFrame(request_data)
    request_dataframe = request_dataframe[data_columns_order]
    request_dataframe_encoded = encode_categorical_features(
        request_dataframe, mapping_unique_values
    )

    prediction_proba = calibrated_model.predict_proba(request_dataframe_encoded)[0, 1]
    prediction_class = 1 if prediction_proba > threshold else 0
    prediction = {
        "probability": prediction_proba,
        "class": mapping_label[prediction_class],
    }

    return jsonify(prediction)


if __name__ == "__main__":
    app.run(debug=True)
