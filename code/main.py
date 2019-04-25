import logging
import os

import numpy as np
import tensorflow as tf
import yaml
from utils import concatenate_dataset, initialize_logger, tensorflow_config

from dataset_handling import pad_per_batch, read_dataset, read_dev_dataset
from model import create_bilstm, create_stacked_bilstm, eval_model, train_model
from preprocessing import bies_format


if __name__ == "__main__":
    # Configure logger to change the message format
    initialize_logger()
    # Allow tensorflow to use memory gpu
    tensorflow_config()

    # Load our config file
    config_file_path = os.path.join(os.getcwd(), 'config.yaml')
    config_file = open(config_file_path)
    config_params = yaml.load(config_file)

    # Read in the dataset
    dataset_name = 'pku_training.utf8'
    training_file = os.path.join(os.getcwd(), dataset_name)
    logging.info(f'Working on dataset in {training_file}')

    # Create input file (with no space), as well as the 'BIES' format
    input_file_name, labels_file_name = bies_format(training_file)

    # Get the max_length attribute for padding use
    MAX_LENGTH = config_params['max_length']

    # Read the training dataset padded
    train_x, train_x_bi, train_y, vocab_dict = read_dataset(
        input_file_name, labels_file_name, MAX_LENGTH)

    # will be used for masking of padding
    pad_mask = np.sign(train_x)

    # Create Unstacked BiLSTM with 'GloVe' pretrained embeddings
    model = create_bilstm(config_params, vocab_dict)

    # For training from the last model saved
    # model_path = os.path.join(os.getcwd(), 'models', 'model-07.h5')
    # model = tf.keras.models.load_model(model_path)

    # Load configuration parameters
    epochs = config_params['epochs']
    batch_size = config_params['batch_size']

    # Get vocabulary_dict for later
    vocab_dict_json = os.path.join(os.getcwd(), 'vocab_dict.json')

    # In a try except, so in order to save model
    # if I wanted to interrupt training anytime
    try:
        # Creating samples array to reduce num of parameters as per method
        samples = [train_x, train_x_bi]
        # Train the model
        history = train_model(model, samples, train_y, epochs,
                              batch_size, pad_mask)
    except KeyboardInterrupt:
        # Save the model in case of keyboard interruption
        model_path = os.path.join(os.getcwd(), 'models', 'model.h5')
        model_weights_path = model_path.replace('model', 'model_weights')

        model.save(model_path)
        model.save_weights(model_weights_path)

        logging.info("Model is saved & process due to process termination.")
