import os
from datetime import datetime
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner


from google.cloud import storage

client = storage.Client()


@task(name="Data Loading")
def load_data(filename):
    """
    Loads data
    """
    df = pd.read_csv(filename, sep=',')

    # drop columns which are not required for training
    df = df.drop(['instant', 'dteday', 'yr', 'casual', 'registered'], axis=1)

    return df


@task(name="Data Splitting")
def split_data(df):
    """
    Splits data into train and valid
    """
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df[['season', 'weekday']])
    return df_train, df_val


@task(name="Prepare Dictionaries")
def prepare_dictionaries(df: pd.DataFrame):
    """
    Prepares dictionaries
    """
    features = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    return df[features].to_dict(orient='records')


@task(name="Train Random Forest Model")
def train_model_rf_search(dict_train, dict_val, y_train, y_val, model_search_iterations, data_path):
    """
    Trains RF model
    """
    mlflow.sklearn.autolog()

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "rf")
            mlflow.log_param("train_data", data_path)

            pipeline = make_pipeline(DictVectorizer(), RandomForestRegressor(**params, n_jobs=-1))

            pipeline.fit(dict_train, y_train)

            y_pred = pipeline.predict(dict_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            r2score = r2_score(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2score)

            mlflow.sklearn.log_model(pipeline, artifact_path="models")

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'n_estimators': scope.int(hp.uniform('n_estimators', 10, 150)),
        'max_depth': scope.int(hp.uniform('max_depth', 1, 40)),
        'min_samples_leaf': scope.int(hp.uniform('min_samples_leaf', 1, 10)),
        'min_samples_split': scope.int(hp.uniform('min_samples_split', 2, 10)),
        'random_state': 42,
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    best_result = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=model_search_iterations, trials=Trials(), rstate=rstate)
    return


@task(name="Register best model")
def register_best_model(tracking_uri, experiment_name, model_registry_name):
    """
    Registers the best model
    """
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    best_run = client.search_runs(experiment_ids=experiment.experiment_id, run_view_type=ViewType.ACTIVE_ONLY, max_results=1, order_by=["metrics.rmse ASC"])[0]

    # register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    model_rmse = best_run.data.metrics['rmse']
    model_details = mlflow.register_model(model_uri=model_uri, name=model_registry_name)

    date = datetime.today().date()

    # transition of our best model in "Production"
    client.transition_model_version_stage(name=model_details.name, version=model_details.version, stage="Production", archive_existing_versions=True)

    client.update_model_version(name=model_details.name, version=model_details.version, description=f"The model version {model_details.version} was transitioned to Production on {date}")

    client.update_registered_model(name=model_details.name, description=f"Current model version in production: {model_details.version}, rmse: {model_rmse}")
