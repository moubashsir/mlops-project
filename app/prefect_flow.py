from prefect import flow, get_run_logger


from prefect_tasks import *

MODEL_REGISTRY_NAME = os.getenv("EXPERIMENT_NAME", "house-price-prediction-model")
MODEL_SEARCH_ITERATIONS = int(os.getenv("MODEL_SEARCH_ITERATIONS", "60"))
TRACKING_SERVER_HOST = "34.125.100.233"
TRACKING_SERVER_PORT = "5000"
DATA_PATH = "/home/mouba/data/day.csv"
EXPERIMENT_NAME = "bike-sharing-regression"





@flow(name="Prefect Cloud Quickstart")
def quickstart_flow():
    logger = get_run_logger()
    logger.warning("Local quickstart flow is running!")

if __name__ == "__main__":
    quickstart_flow()

