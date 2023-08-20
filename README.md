# Washington D.C Capital Bikeshare - Predicting number of bike bookings

## Objective

Applying learnings from [MLOps ZoomCamp](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/) course to a real-life ML problem using MLOps best practices. 

## Problem definition
[Bikeshare bike booking data set](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) contains the hourly and daily count of rental bikes between years 2011 and 2012 in Capital bikeshare system in Washington, DC with the corresponding weather and seasonal information. The aim is to build a prediction model which predicts the number of bookings in a day based on day information (like which day of the week, is it a holiday, season etc) and weather information of the day (like temperatue, humidity, windspeed etc.)

## MLOps Architecture
![plot](/images/Architecture.jpg)

## Applied Technologies

| Technology Used        | Scope         |
| ------------- | ------------- |
| Google Compute Engine      | Used VM instance to do all computation |
| Google Cloud Storage Bucket      | Used as data store and artifact store for MLflow experiments      |
| Docker | Application containerization |
| Docket-compose | Multi-container Docker applications definition and running |
| PostgreSQL | Used PostgreSQL database on GCP as database for tracking MLFlow experiments      |
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

All components of MLOps are developed on a Google VM instance. 

## Model development 
You can click [here](/app/prefect_flows.py) to see the data modeling's part of the project. What does the code do ?

1. It retrieves the data, since it is a static data and there is no update to data after 2012, I have placed the data in a Google Cloud storage bucket
2. Then it creates training and validation data and fits a DictVectorizer
3. It tunes hyperparameters from a Random Forest classifier, and logs every metrics in MLflow runs.
4. It registers the model (best one) as the production one in the registry if it is a better one than the current model in production

## Scheduled Model training
Once the model was trained, the entire model development pipeline, right from downloading the data, to finding the best hyperparameter and then fitting a model then finding the best model and moving it to production is automated using Prefect Workflow Orchestrator. The prefect server is dockerized to run in a separate container. The configuration of the container can be found [here](/docker-compose.yml). The Prefect flow script can be found [here](/app/prefect_flows.py)

## Prediction Service
Flask is used as a web-server. The web-service module does the following:

1. Load the best model (scikit-learn pipeline of DictVectorizer and trained model) from model registry if model registry is up and running, else it picks the model from GCS bucket based on RUN_ID
2. Read the provided input-json and sends prediction

Prediction service is dockerized to run on a separate container.

Prediction service script is [here](/app/web_service.py)

## Model Monitoring

### Batch Prediction as a simulated traffic:
Using batach prediction to create a simulation of how an actual traffic data look like. The predictions are then collected and used in monitoring dashboard. Batch prediction script is [here](/app/batch_prediction.py)

### Monitoring 

Evidently is used for model monitoring along with FastAPI. Prediction drift and data drift is used from Evidently to monitor the model. Monitoring service is dockerized to run on a separate container. You can see the code [here](/monitoring/app.py). 

## Starting the services

### Initial setup

1. A VM intance should be created along with a database (PstgreSQL) on GCP following this [article](https://kargarisaac.github.io/blog/mlops/data%20engineering/2022/06/15/MLFlow-on-GCP.html)
2. Authenticating Google Cloud Login: If you are running the code from a GCP VM then you can use this [link](https://cloud.google.com/sdk/gcloud/reference/auth/application-default/login) to obtain access credentials for your user account. If you are running on a local machine, you need to install "gcloud" SKD and run the same command from gcloud terminal. Alternatively, you can generate GOOGLE_APPLICATION_CREDENTIALS json file by following this [link](https://cloud.google.com/docs/authentication/client-libraries) and then set the GOOGLE_APPLICATION_CREDENTIALS environment variable using the below command.
```
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/client_secret.json'
```
3. Ensure you have docker and docker-compose set up on you machine by followinng the environment setup video [here](https://www.youtube.com/watch?v=IXSiYkP23zo&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK)
4. Next, clone the repository
```
git clone https://github.com/moubashsir/mlops-project
```
5. Start your flow server
```
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@{POSTGRES_SERVER_HOST}:{POSTGRES_SERVER_PORT}/${POSTGRES_DB} --artifacts-destination gs://${GCS_BUCKET_NAME}/{FOLDER_NAME}
```

Once run, mlflow server will be available at port 5000

6. Ensure that docker and docker-compose are setup and user is added to docker group so that commands can be run with "sudo"
7. In docker-compose.yml file change the environment variable TRACKING_SERVER_HOST and give correct host name as per where you server is hosted. You can also change the variable MODEL_SEARCH_ITERATIONS based on your liking. From the "mlops-project" folder run
```
docker-compose build
```
8. Once the containers are built, run
```
docker-compose up
```
9. This will start 3 services, at below ports

| Service       | Port         | Interface | Description |
| ------------- | ------------- | ------------- | ------------- |
| prefect-server     | 4200 | 127.0.0.1 | Training workflow orchestration | 
| prediction-service | 9696 | 127.0.0.1 | Prediction service |
| monitoring-service | 8001 | 127.0.0.1 | Model monitoring service | 

10. Once the containers are running You can test the services
11. For testing prediction service run below command in a separate terminal. It will return the predicted count of bike booking based on input dictionary provided in the prediction.py file
```
python /app/prediction.py
``` 
12. To see the evidently dashboard and monitor model performance, open the below location in you browser:
```
http://localhost:8001/get_dashboard
```
13. You can schedule a training flow using prefect once you have created a prefect workpool and started the worker by runniing the below
```
docker exec -t prefect-server prefect deploy prefect_flows.py:main_flow -p training-pool
```
Or alternatively, you can directly run the retraining scrip.
```
docker exec -t prefect-server python /app/prefect_flows.py
```

On my side the deployment is scheduled to run on every Monday 5:00 AM
14. MLOps pipeline can be stopped by running 
```
docker-compose down
```

## Self assessment of the project

* Problem descriptiopn:
    * 2 points: The problem is well described and it's clear what the problem the project solves
* Cloud
    * 2 points: The project is developed on the cloud OR uses localstack (or similar tool) OR the project is deployed to Kubernetes or similar container management platforms
* Experiment tracking and model registry
    * 4 points: Both experiment tracking and model registry are used
* Workflow orchestration
    * 4 points: Fully deployed workflow
* Model deployment
    * 4 points: The model deployment code is containerized and could be deployed to cloud or special tools for model deployment are used
* Model monitoring
    * 2 points: Basic model monitoring that calculates and reports metrics
* Reproducibility
    * 4 points: Instructions are clear, it's easy to run the code, and it works. The versions for all the dependencies are specified
* Best practices
    * There are unit tests (1 point)
    * There is an integration test (1 point)
    * Linter and/or code formatter are used (1 point)