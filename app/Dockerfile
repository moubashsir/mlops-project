#FROM python:3.9.12-slim

#RUN pip install -U pip
#RUN pip install pipenv 

#WORKDIR /app

#COPY [ "Pipfile", "Pipfile.lock", "./" ]

#RUN pipenv install --system --deploy

#COPY [ "web_service.py", "./" ]

#EXPOSE 9696

#ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "web_service:app" ]

FROM prefecthq/prefect:latest 

RUN pip install -U pip
RUN pip install -U prefect

WORKDIR /app 

COPY . .

RUN pip install -r requirements.txt

CMD python prefect_flows.py