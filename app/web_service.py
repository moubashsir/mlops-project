import os

import mlflow
from flask import Flask, request, jsonify

from google.cloud import storage

client = storage.Client()

TRACKING_SERVER_HOST = os.getenv("TRACKING_SERVER_HOST", "localhost")
TRACKING_SERVER_PORT = os.getenv("TRACKING_SERVER_PORT", "5000")
MODEL_NAME = os.getenv("MODEL_NAME", "random-forest-regressor")
RUN_ID = os.getenv("RUN_ID", "79934c79a98f4932aade316cce6e61a0")
TRACKING_URI = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"
MLFLOW_ENABLED = False #if set false it takes best model based on RUN ID from GCS bucket

def load_model_from_registry():
    """
    Loads the ML model either from ML flow registry or from GCS bucket
    """
    tracking_uri = TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{MODEL_NAME}/latest"

    gcs_bucket = f"gs://mlflow-assignment-mj/bike-sharing-prediction/3/{RUN_ID}/artifacts/model"

    if MLFLOW_ENABLED:
        print("Model loaded from registry")
        return mlflow.pyfunc.load_model(model_uri)
    else:
        print("Model loaded from GCS bucket")
        return mlflow.pyfunc.load_model(gcs_bucket)


def predict(features):
    """
    predict the count based on input features
    """
    model = load_model_from_registry()
    preds = model.predict(features)
    return float(preds[0])


app = Flask('ride-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Flask App to get predictions
    """
    bike_data = request.get_json()

    pred = predict(bike_data)

    result = {
        'count': pred,
        #'model_version': load_model_from_registry
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
