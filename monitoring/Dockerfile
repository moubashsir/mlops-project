FROM python:3.9.12-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY . .

EXPOSE 8001

CMD uvicorn app:app --host 0.0.0.0 --port 8001