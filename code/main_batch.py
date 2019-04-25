import logging
import os

import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm
from utils import initialize_logger, tensorflow_config

from dataset_handling import pad_per_batch, process_dataset
from model import create_bilstm
from preprocessing import bies_format, generate_labels_dict
from tensorflow.keras.callbacks import TensorBoard


if __name__ == "__main__":
    # Configure logger to change the message format
    initialize_logger()
    # Allow tensorflow to use memory gpu
    tensorflow_config()

    # Read the configuration file
    config_file_path = os.path.join(os.getcwd(), 'code', 'config.yaml')
    config_file = open(config_file_path)
    config_params = yaml.load(config_file)

    # for Readability and logging purposes
    dataset_name = 'cityu_msr_pku_concat.utf8'
    training_file = os.path.join(os.getcwd(), 'code', dataset_name)
    logging.info(f'Working on dataset in {training_file}')

    # Create 'BIES' format for training dataset file
    input_file_name, labels_file_name = bies_format(training_file)

    # Load all the samples, equivalent labels (Unpadded), vocab_dict
    (train_x, train_x_bi,
     train_y, vocab_dict) = process_dataset(input_file_name, labels_file_name)

    logging.info(f'Vocabulary Size: {len(vocab_dict)}')

    # Create the Unstacked Bi-LSTM with 'GloVe' pretrained embeddings
    model = create_bilstm(config_params, vocab_dict)

    # Load configuration parameters
    epochs = config_params['epochs']
    batch_size = config_params['batch_size']

    # Set where model and its weights will be saved
    model_path = os.path.join(os.getcwd(), 'code', 'models', 'model.h5')
    model_weights_path = model_path.replace('model.h5', 'model_weights.h5')

    # For logging purposes during training
    n_iterations = int(np.ceil(len(train_x)/batch_size))

    # Training loop
    epochs = 1
    for epoch in tqdm(range(1, epochs + 1), desc='Training model'):
        print(f'\nEpoch: {epoch}')
        mb = 0
        epoch_loss, epoch_acc = 0., 0.
        # Train on batch, as we want to pad data as (max sample len) per batch
        # To avoid the loss of any information
        for pad_train_x, pad_train_x_bi, pad_train_y in pad_per_batch(train_x,
                                                                      train_x_bi,
                                                                      train_y,
                                                                      batch_size):
            # For showing completion rate as per batch
            mb += 1
            # Train on batches of data
            loss, acc = model.train_on_batch(
                [pad_train_x, pad_train_x_bi], pad_train_y)
            # Log values as per epoch
            epoch_loss += loss
            epoch_acc += acc

            # For printing purposes as per batch
            print_loss = round(epoch_loss/mb, 4)
            print_acc = round((epoch_acc/mb)*100, 4)
            completion_rate = round(100.*mb/n_iterations, 2)
            print(
                f"Completion Rate: {completion_rate} % || Train Loss: {print_loss} || Train Accuracy: {print_acc} %", end="\r")
        # Logging as per epoch
        epoch_loss /= n_iterations
        epoch_acc /= n_iterations
        print(
            f"\nAverage Per Epoch >> Train Loss: {np.round(epoch_loss, 4)} || Train Accuracy: {np.round(epoch_acc, 4)} %")
        print('_________'*10)

        # Save model each epoch
        model.save(model_path)
        model.save_weights(model_weights_path)
        logging.info(f'Model and its weights are saved for epoch {epoch}')

        # For model evaluation, check `predict.py`
