FROM python:3.10.14-bookworm

RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip

COPY drug_generator /app/drug_generator
COPY tests /app/tests
COPY main.py /app

WORKDIR /app

RUN mkdir /app/app_data

RUN chmod -R 777 /app/app_data

ENV PYTHONPATH=${PYTHONPATH}:/app/drug_generator

RUN pip install --no-cache-dir -r drug_generator/requirements.txt

EXPOSE 8000

RUN chmod +x drug_generator/train_pipeline.py
RUN chmod +x drug_generator/predict.py

RUN chmod +x /app/main.py
CMD pip install -e .