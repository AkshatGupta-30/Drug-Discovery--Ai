import os
import pandas as pd
import tensorflow as tf
from drug_generator.config import config

def load_dataset(file_name):
    file_path = os.path.join(config.DATAPATH,file_name)
    data=pd.read_csv(file_path)
    return data


def save_model(model_to_save):
    save_path = os.path.join(config.SAVED_MODEL_PATH, config.MODEL_FILE_NAME)
    model_to_save.save(save_path)
    print("Model saved at", save_path)

def load_nn_model():
    save_path = os.path.join(config.SAVED_MODEL_PATH, config.MODEL_FILE_NAME)
    pretrained_model = tf.keras.models.load_model(save_path)
    return pretrained_model