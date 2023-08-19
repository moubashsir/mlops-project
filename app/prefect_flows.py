from prefect_tasks import *

TRACKING_SERVER_HOST = os.getenv("TRACKING_SERVER_HOST", "localhost")
TRACKING_SERVER_PORT = os.getenv("TRACKING_SERVER_PORT", "5000")
DATA_PATH = os.getenv("DATA_PATH", "https://storage.googleapis.com/mlflow-assignment-mj/training_data/day.csv")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "bike-sharing-regression")
MODEL_REGISTRY_NAME = os.getenv("MODEL_REGISTRY_NAME", "random-forest-regressor")
MODEL_SEARCH_ITERATIONS = int(os.getenv("MODEL_SEARCH_ITERATIONS", "10"))

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

    train_model_rf_search(dict_train, dict_val, y_train, y_val, MODEL_SEARCH_ITERATIONS, DATA_PATH)

    register_best_model(tracking_uri, EXPERIMENT_NAME, MODEL_REGISTRY_NAME)
    logger.info("Successfully executed our flow !!!")


if __name__ == "__main__":
    main()
