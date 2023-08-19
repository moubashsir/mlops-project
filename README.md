# Washington D.C Capital Bikeshare - Predicting number of bike bookings

## Objective

Applying learnings from [MLOps ZoomCamp](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/) course to a real-life ML problem using MLOps best practices. 

## Problem definition
[Bikeshare bike booking data set](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) contains the hourly and daily count of rental bikes between years 2011 and 2012 in Capital bikeshare system in Washington, DC with the corresponding weather and seasonal information. The aim is to build a prediction model which predicts the count of booking based on day information (like which day of the week, is it a holiday, what season is it?) and weather information of the day (like temperatue, humidity, windspeed etc.)

## MLOps Architecture
insert the architecture diagram

## Applied Technologies

| Technology Used        | Scope         |
| ------------- | ------------- |
| Google Compute Engine      | Used VM instance to do all computation |
| Google Cloud Storage Bucket      | Used as data store and artifact store for MLflow experiments      |
| Docker | Application containerization |
| Docket-compose | Multi-container Docker applications definition and running |
| PostgreSQL | Used PostgreSQL database on GCP as database for tracking MLFlow experiments      |
| Jupyter Notebooks | EDA and initial model devel |
| Jupyter Notebooks | EDA and initial model devel |
| MLFlow | Experiment Tracking and model registry |
| Prefect | Workflow orchestration |
| scikit-learn | ML model training |
| Flask | Web server |
| FastAPI | Web server |
| EvidentlyAI | ML model monitoring |
| pytest | Unit and integration testing | 
| pylint | Python static code analysis |
| black | Python code formatting |
| isort | Python import sorting |

## Cloud Services
During the implementation of this project, I have used some cloud services such as Google Compute Services, PostgreSQL on GCP, Google Cloud Storage and Prefect Cloud.

## Model development 
You can click here to see the data modeling's part of our project. What does the code do ?

1. It retrieves the data, since it is a static data and there is no update to data after 2012, I have placed the data in a Google Cloud storage bucket
2. Then it creates training and validation data and fits a DictVectorizer
3. It tunes hyperparameters from aa Random Forest classifier, and logs every metrics in MLflow runs.
4. It registers the model (best one) as the production one in the registry if it is a better one than the current model in production

## Scheduled Model training
Once the model was trained, the entire model development pipeline, right from downloading the data, to finding the best hyperparameter and then fitting a model then finding the best model and moving it to production is automated using Prefect Workflow Orchestrator. On my side, the training workflow is automated to run every Monday at 5:00 AM. The flow can be run either by directly starting a local Prefect Server or by creating a Prefect Cloud account and following the steps described [here](https://docs.prefect.io/2.11.3/cloud/cloud-quickstart/)

## Prediction Service
Flask is used as a web-server. The web-service module does the following:

1. Load the best model (scikit-learn pipeline of DictVectorizer and trained model) from model registry if model registry is up and running, else it picks the model from GCS bucket based on RUN_ID
2. Read the provided input-json and sends prediction

## Model Monitoring

### Batch Prediction:
First a batch prediction module is created to get predictions from the model to use it in monitoring frame-work, for now the batch-prediction module takes input from a simulated (synthetic data created by adding small noise in the original data set) data placed in a GCS bucket and the outputs a csv file

### Monitoring 

Evidently is used for model monitoring along with FastAPI. Prediction drift and data drift is used from Evidently to monitor the model

## 