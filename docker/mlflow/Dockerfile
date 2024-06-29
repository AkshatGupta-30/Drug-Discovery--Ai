FROM python:3.10.14-bookworm

RUN pip3 install mlflow==2.12.2
RUN pip3 install mysql-connector-python
RUN pip3 install mysqlclient
RUN pip3 install pymysql

EXPOSE 5000

RUN mkdir -p /mlruns
RUN chmod 777 /mlruns

# ENTRYPOINT ["mlflow", "server"]
# CMD ["--port", "5000", "--backend-store-uri", "mysql://mlflow_user:mlflow@localhost:3306/db_mlflow", "--default-artifact-root", "/mlruns/", "--host", "0.0.0.0"]

CMD ["mlflow", "server", "--backend-store-uri", "mysql+pymysql://mlflow:mlflow@mysql:3306/db_mlflow", "--artifacts-destination", "/mlartifacts", "--serve-artifacts", "-h 0.0.0.0", "-p 5000"]