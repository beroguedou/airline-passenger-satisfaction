import pandas as pd
from flask import Flask, jsonify, request, wrappers
from utils import get_catalog_and_params, inference

catalog, parameters = get_catalog_and_params()
calibrated_model = catalog.load("trained_calibrated_model")
data_columns_order = catalog.load("data_columns_order")
mapping_unique_values = catalog.load("mapping_unique_values")
mapping_label = mapping_unique_values["satisfaction"]
del mapping_unique_values["satisfaction"]

threshold = parameters["threshold"]

app = Flask(__name__)


@app.route("/predict/satisfaction", methods=["POST"])
def predict_satisfaction() -> wrappers.Response:
    request_data = request.json
    request_data = {k: [v] for (k, v) in request_data.items()}
    request_dataframe = pd.DataFrame(request_data)
    prediction = inference(
        request_dataframe,
        mapping_unique_values,
        data_columns_order,
        calibrated_model,
        mapping_label,
        threshold,
    )
    return jsonify(prediction)


if __name__ == "__main__":
    app.run(debug=True, port=5001, host="0.0.0.0")
