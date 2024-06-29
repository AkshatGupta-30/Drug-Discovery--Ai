FROM python:3.10.14-bookworm

RUN pip install --upgrade pip

COPY drug_generator /app/drug_generator
WORKDIR /app

ENV PYTHONPATH=${PYTHONPATH}:/app/drug_generator

RUN pip install -r drug_generator/requirements.txt

RUN chmod +x drug_generator/train_pipeline.py
RUN chmod +x drug_generator/predict.py

ENTRYPOINT [ "python3" ]
CMD [ "drug_generator/train_pipeline.py" ]