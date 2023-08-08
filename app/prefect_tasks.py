import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.feature_extraction import DictVectorizer

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner


from google.cloud import storage
client = storage.Client()

MODEL_REGISTRY_NAME = os.getenv("EXPERIMENT_NAME", "house-price-prediction-model")
MODEL_SEARCH_ITERATIONS = int(os.getenv("MODEL_SEARCH_ITERATIONS", "60"))
TRACKING_SERVER_HOST = "34.16.191.116"
TRACKING_SERVER_PORT = "5000"
DATA_PATH = "~/data/day.csv"
EXPERIMENT_NAME = "bike-sharing-regression"
MODEL_REGISTRY_NAME = "random-forest-regressor"
MODEL_SEARCH_ITERATIONS = 10


@task(name="Data Loading")
def load_data(filename):
    df = pd.read_csv(filename, sep=',')
    
    # drop columns which are not required for training
    df = df.drop(['instant', 'dteday', 'yr', 'casual', 'registered'], axis=1)
    
    return df

@task(name="Data Splitting")
def split_data(df):
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42,
                                        stratify=df[['season', 'weekday']])
    return df_train, df_val


@task(name="Prepare Dictionaries")
def prepare_dictionaries(df: pd.DataFrame):
    features = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
                        'temp', 'atemp', 'hum', 'windspeed']
    return df[features].to_dict(orient='records')


@task(name="Train Random Forest Model")
def train_model_rf_search(dict_train, dict_val, y_train, y_val, model_search_iterations):
    mlflow.sklearn.autolog()

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "rf")
            mlflow.log_param("train_data",DATA_PATH)
            
            pipeline = make_pipeline(
                DictVectorizer(),
                RandomForestRegressor(**params, n_jobs=-1)
            )
            
            pipeline.fit(dict_train, y_train)

            y_pred = pipeline.predict(dict_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            r2score = r2_score(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2score)
                
            mlflow.sklearn.log_model(pipeline, artifact_path="models")

        return {'loss': rmse, 'status': STATUS_OK}


    search_space = {
        'n_estimators' : scope.int(hp.uniform('n_estimators',10,150)),
        'max_depth' : scope.int(hp.uniform('max_depth',1,40)),
        'min_samples_leaf' : scope.int(hp.uniform('min_samples_leaf',1,10)),
        'min_samples_split' : scope.int(hp.uniform('min_samples_split',2,10)),
        'random_state' : 42
    }
    
    rstate = np.random.default_rng(42)  # for reproducible results
    best_result =  fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=model_search_iterations,
        trials=Trials(),
        rstate=rstate
    )
    return


@task(name="Register best model")
def register_best_model(tracking_uri, experiment_name, model_registry_name):
    
    client = MlflowClient(tracking_uri=tracking_uri)
    
    experiment = client.get_experiment_by_name(experiment_name)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"]
    )[0]
    
    # register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    model_rmse = best_run.data.metrics['rmse']
    model_details = mlflow.register_model(model_uri=model_uri, name=model_registry_name)

    date = datetime.today().date()
    
    # transition of our best model in "Production"
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    client.update_model_version(
        name=model_details.name,
        version=model_details.version,
        description=f"The model version {model_details.version} was transitioned to Production on {date}"
    )
    
    client.update_registered_model(
        name=model_details.name,
        description=f"Current model version in production: {model_details.version}, rmse: {model_rmse}"
    )


@flow(task_runner=SequentialTaskRunner())
def main():
    """
    Executes the training workflow
    """
    tracking_uri = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger = get_run_logger()

    df = load_data(DATA_PATH)
    df_train, df_val = split_data(df)

    dict_train = prepare_dictionaries(df_train)
    dict_val = prepare_dictionaries(df_val)
    
    target = 'cnt'
    y_train = df_train[target].values
    y_val = df_val[target].values
    
    train_model_rf_search(dict_train, dict_val, y_train, y_val, MODEL_SEARCH_ITERATIONS)
    
    register_best_model(tracking_uri, EXPERIMENT_NAME, MODEL_REGISTRY_NAME)
    logger.info("Successfully executed our flow !!!")

if __name__ == "__main__":
    main()