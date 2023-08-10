

import os

import mlflow
from flask import Flask, request, jsonify

from google.cloud import storage
client = storage.Client()

TRACKING_SERVER_HOST = "34.125.106.170"
TRACKING_SERVER_PORT = "5000"
MODEL_NAME = "random-forest-regressor"
TRACKING_URI=f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"

def load_model_from_registry():
    """
    Loads the ML model from the MLFlow registry
    """
    tracking_uri = TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{MODEL_NAME}/latest"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded from registry")

    return loaded_model

def predict(features):
    model = load_model_from_registry()
    preds = model.predict(features)
    return float(preds[0])


app = Flask('ride-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    bike_data = request.get_json()

    pred = predict(bike_data)

    result = {
        'count': pred,
        #'model_version': load_model_from_registry
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

