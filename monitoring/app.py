import os
import warnings
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric


warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "scored_reference.csv")
CURRENT_DATA_PATH = os.getenv("CURRENT_DATA_PATH", "scored_current.csv")
REPORT_PATH = os.getenv("REPORT_PATH", "./dashboards/data_drift.html")

app = FastAPI()


def load_data(filename):
    """Load data"""
    df = pd.read_csv(filename, sep=',')

    req_column = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'prediction']

    return df[req_column]


def create_dashboard():
    """create dashboard"""

    ref_data_sample = load_data(REFERENCE_DATA_PATH)
    prod_data_sample = load_data(CURRENT_DATA_PATH)
    num_features = ['temp', 'atemp', 'hum', 'windspeed']
    cat_features = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']

    column_mapping = ColumnMapping(prediction='prediction', numerical_features=num_features, categorical_features=cat_features, target=None)

    report = Report(metrics=[ColumnDriftMetric(column_name='prediction'), DatasetDriftMetric(), DatasetMissingValuesMetric()])

    report.run(reference_data=ref_data_sample, current_data=prod_data_sample, column_mapping=column_mapping)

    report.save_html(REPORT_PATH)


@app.get("/get_dashboard")
async def data_drift():
    """api to get dashboard"""
    create_dashboard()
    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        dashboard = file.read()

    return HTMLResponse(content=dashboard, status_code=200)
