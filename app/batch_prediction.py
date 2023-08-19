import os

import numpy as np
import pandas as pd
import mlflow

from google.cloud import storage
client = storage.Client()

TRACKING_SERVER_HOST = "34.125.104.8"
TRACKING_SERVER_PORT = "5000"
MODEL_NAME = "random-forest-regressor"
TRACKING_URI=f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"
MLFLOW_ENABLED = False
RUN_ID = "79934c79a98f4932aade316cce6e61a0"
DATA_PATH = "https://storage.googleapis.com/mlflow-assignment-mj/training_data/day.csv"


def load_data(filename):
    """
    Loads data
    """
    df = pd.read_csv(filename, sep=',')
    
    # drop columns which are not required for training
    df = df.drop(['instant', 'dteday', 'yr', 'casual', 'registered'], axis=1)
    
    return df


def prepare_dictionaries(df: pd.DataFrame):
    features = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
                        'temp', 'atemp', 'hum', 'windspeed']
    return df[features].to_dict(orient='records')

def create_synthetic_data(df):
    """
    Creates dummy data for batch prediction
    """
    current_data = pd.DataFrame()
    
    current_data['season'] = np.random.permutation(df['season'].values)
    current_data['mnth'] = np.random.permutation(df['mnth'].values)
    current_data['holiday'] = np.random.permutation(df['holiday'].values)
    current_data['weekday'] = np.random.permutation(df['weekday'].values)
    current_data['workingday'] = np.random.permutation(df['workingday'].values)
    current_data['weathersit'] = np.random.permutation(df['weathersit'].values)
    current_data['temp'] = np.random.permutation(df['temp'].values)
    current_data['atemp'] = np.random.permutation(df['atemp'].values)
    current_data['hum'] = np.random.permutation(df['hum'].values)
    current_data['windspeed'] = np.random.permutation(df['windspeed'].values)
    return current_data

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


def main():
    """
    Prediction on reference and current data
    """
    
    raw_ref_data = load_data(DATA_PATH)
    current_data = create_synthetic_data(raw_ref_data)
    
    loaded_model = load_model_from_registry()
    
    dict_reference = prepare_dictionaries(raw_ref_data)
    pred_reference = loaded_model.predict(dict_reference)
    raw_ref_data['prediction'] = pred_reference
    
    dict_current = prepare_dictionaries(current_data)
    pred_current = loaded_model.predict(dict_current)
    current_data['prediction'] = pred_current

    raw_ref_data.to_csv("~/mlops-project/monitoring/scored_reference.csv")
    current_data.to_csv("~/mlops-project/monitoring/scored_current.csv")
    
if __name__ == "__main__":
    main()


