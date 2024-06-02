import pandas as pd
import numpy as np
import random

from keras.models import Model
from keras.layers import Input

from drug_generator.config import config
from drug_generator.preprocessing.data_management import load_dataset, load_nn_model
from drug_generator.preprocessing import preprocessors as pp

loaded_model = load_nn_model(config.MODEL_FILE_NAME)

cv_data = load_dataset(config.TEST_FILE)
preprocess = pp.preprocess_data()
preprocess.fit()
X_cv, _ = preprocess.transform(cv_data['X'], cv_data['Y'])


def inference_encoder():
    enc_input = loaded_model.input[0]
    enc_embedding = loaded_model.layers[2](enc_input)
    enc_lstm_layer = loaded_model.layers[4]
    enc_lstm_output,enc_last_hidden_state,enc_last_cell_state = enc_lstm_layer(inputs=enc_embedding)
    inf_enc = Model(
        inputs=enc_input,
        outputs=[enc_lstm_output,enc_last_hidden_state,enc_last_cell_state]
    )
    return inf_enc


def inference_decoder():
    dec_input = loaded_model.input[1]
    another_dec_input = Input(
        shape=(
            config.MAX_INPUT_SEQUENCE_LEN,
            (len(config.BASE_VOCABULARY)+2)//2
        )
    )
    dec_initial_hidden_state = Input(shape=((len(config.BASE_VOCABULARY)+2)//2,))
    dec_initial_cell_state = Input(shape=((len(config.BASE_VOCABULARY)+2)//2,)) 
    dec_embedding = loaded_model.layers[3](dec_input)
    dec_lstm_layer = loaded_model.layers[5]
    dec_lstm_output,dec_last_hidden_state,dec_last_cell_state = dec_lstm_layer(
        inputs=dec_embedding,
        initial_state=[dec_initial_hidden_state,dec_initial_cell_state]
    )
    dec_enc_attn_seq = loaded_model.layers[6]([dec_lstm_output,another_dec_input])
    dec_dense_input = loaded_model.layers[7]([dec_lstm_output,dec_enc_attn_seq])
    dec_output = loaded_model.layers[8](dec_dense_input)

    inf_dec = Model(
        inputs=[
            dec_input,another_dec_input,
            dec_initial_hidden_state,
            dec_initial_cell_state
        ],
        outputs=[
            dec_output,
            dec_last_hidden_state,
            dec_last_cell_state
        ]
    )
    
    return inf_dec



def generate_molecule(enc_inp_sequence,batch_size):
    inf_enc = inference_encoder()
    states = inf_enc.predict(enc_inp_sequence)
    enc_output = states[0]
    states.pop(0)
    gen_sequence = np.array([[config.INPUT_VOCABULARY.index("<")]*batch_size])
    stop_generation = False
    generated_molecule= str()
    inf_dec = inference_decoder()
    
    while not stop_generation:
        dec_output,dec_last_hidden_state,dec_last_cell_state = inf_dec.predict(
            [gen_sequence,enc_output]
            + states
        )
        nxt_gen_char_idx = np.argmax(dec_output[0,-1,:])
        nxt_gen_char = config.OUTPUT_VOCABULARY[nxt_gen_char_idx]
        generated_molecule += nxt_gen_char
        if (nxt_gen_char == "\n") or (len(generated_molecule) == config.MAX_OUTPUT_SEQUENCE_LEN):
            stop_generation = True
            
        gen_sequence = np.array([[nxt_gen_char_idx]*batch_size])
        states = [dec_last_hidden_state,dec_last_cell_state]

    return {"Generated Valid Molecule" : generated_molecule}

if __name__ == '__main__':
    single_input = X_cv[random.randint(0, X_cv.shape[0]), :]
    single_input = single_input.reshape(1, single_input.shape[0])
    gen_mol = generate_molecule(single_input, 1)
    print(gen_mol)