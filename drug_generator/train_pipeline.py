import os

import mlflow.keras.autologging
import pandas as pd
import numpy as numpy
from keras.utils import to_categorical
import mlflow
import mlflow.keras
import mlflow.tensorflow

from drug_generator.config import config
from drug_generator.preprocessing.data_management import load_dataset, save_model, load_nn_model
import drug_generator.preprocessing.preprocessors as pp
import drug_generator.pipeline as pl

def run_training():
    training_data = load_dataset(config.TRAIN_FILE)
    cv_data = load_dataset(config.TEST_FILE)

    preprocess = pp.preprocess_data()
    preprocess.fit()
    X_train, Y_train = preprocess.transform(training_data["X"], training_data["Y"])
    X_cv, Y_cv = preprocess.transform(cv_data["X"], cv_data["Y"])

    def training_data_generator(mb_size,epochs):
        for _ in range(epochs):
            for i in range(X_train.shape[0]//mb_size):
                yield (
                    [
                        X_train[i * mb_size : (i+1) * mb_size],
                        X_train[i * mb_size : (i+1) * mb_size]
                    ],
                    to_categorical(
                        Y_train[i * mb_size : (i+1) * mb_size],
                        num_classes=len(config.BASE_VOCABULARY)+2
                    )
                )
    
    def cv_data_generator(mb_size,epochs):
        for _ in range(epochs):
            for i in range(X_cv.shape[0]//mb_size):
                yield (
                    [
                        X_cv[i * mb_size : (i+1) * mb_size],
                        X_cv[i * mb_size : (i+1) * mb_size]
                    ],
                    to_categorical(
                        Y_cv[i * mb_size : (i+1) * mb_size],
                        num_classes=len(config.BASE_VOCABULARY)+2
                    )
                )

    drug_discovery_model = pl.encoder_decoder_with_attn_mech()
    history = drug_discovery_model.fit(
        training_data_generator(1000,25),
        epochs=25,
        steps_per_epoch=1000,
        validation_data=cv_data_generator(1711,25),
        validation_steps=18
    )
    save_model(drug_discovery_model)
    return history

def mlflow_logs(model,model_history,name):
    with mlflow.start_run(run_name=name) as run:
        exp_run_id = run.info.run_id
        mlflow.set_tag("run_id",exp_run_id)
        mlflow.log_metric("Training Accuracy", model_history.history["Accuracy"][-1])
        mlflow.log_metric("Validation Accuracy", model_history.history["val_Accuracy"][-1])
        mlflow.keras.log_model(model,name)
        mlflow.end_run()

if __name__ == "__main__":
    model_history = run_training()
    loaded_model = load_nn_model(config.MODEL_FILE_NAME)
    mlflow.set_experiment("drug_molecule_discovery")
    mlflow_logs(loaded_model,model_history,config.MODEL_FILE_NAME)