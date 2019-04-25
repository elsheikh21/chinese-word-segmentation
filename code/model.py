import json
import logging
import os

import numpy as np
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.layers import (LSTM, Bidirectional, Concatenate, Dense,
                                     Embedding, Input, Masking,
                                     TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tqdm import tqdm
from utils import initialize_logger


def pretrained_embeddings(vocab_dict, embbeddings_file='wiki.zh.vec'):
    '''
    Fetches the pretrained embeddings file from current working directory,
    create a dictionary to have each character and its corresponding vector
    then creates a matrix from this vector, and returns the matrix

    Arguments:
        vocab_dict: dictionary containing indices as per unique uni and bigrams
    Returns:
        embedding_matrix: numpy array of all embeddings as per vec file
    '''
    # first, build index mapping words in the embeddings set, to their embedding vector
    # Gets pretrained embeddings file path
    pretrained_embeddings_path = os.path.join(
        os.getcwd(), 'code', embbeddings_file)
    # Creates a dict
    embeddings_index = dict()
    # Opens the utf-8 encoded file, loops on every line
    # and assign the value as per char
    with open(pretrained_embeddings_path, encoding='utf-8', mode='r') as f:
        for line in tqdm(f, desc='Loading embeddings dict'):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float64')
            embeddings_index[word] = coefs

    # For our vocabulary size
    vocab_size = len(vocab_dict)

    # Create embeddings matrix which will be passed to the model later
    # the matrix size is [vocab size x 300]
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, index in tqdm(vocab_dict.items(),
                            desc='Creating embeddings matrix'):
        if index > vocab_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros
                embedding_matrix[index] = embedding_vector

    logging.info('Pretrained Embeddings are created.')
    # Returns embeddings matrix
    return embedding_matrix


def create_bilstm(config_params, vocab_dict):
    '''
    Creates an unstacked bi-lstm model with inputs shape of max_len
    argument passed,  it does use the loaded pretrained embeddings matrix,
    specifying num of cells as per LSTM, dropouts, and
    recurrent dropouts, finally, takes learning rate, momentum,
    nesterov params to build our optimizer

    Arguments:
        max_len
        vocab_size
        embedding_matrix
        num_cells
        dropout
        rec_dropout
        num_outputs
        init_lr
        momentum
        nesterov
    Returns:
        Model
    '''

    logging.info('Creating Bi-LSTM unstacked model')

    # load model params first from config params passed
    max_len = int(config_params['max_length'])
    num_cells = int(config_params['num_cells'])
    dropout = float(config_params['dropout'])
    rec_dropout = float(config_params['rec_dropout'])
    num_outputs = int(config_params['num_outputs'])
    batch_size = int(config_params['batch_size'])
    init_lr = float(config_params['init_lr'])
    momentum = float(config_params['momentum'])
    nesterov = config_params['nesterov']

    vocab_size = len(vocab_dict)

    # load pretrained embeddings
    embedding_matrix = pretrained_embeddings(vocab_dict)

    # Create the unigrams & bigrams input layers
    inputs_uni = Input(shape=(None,))
    inputs_bi = Input(shape=(None,))

    # Create embeddings as per each layer
    # Not trainable because they are already pretrained
    # Output dim is 300, in order to have more contextual
    # and semantical meaning.
    # Of course masking zeros
    embedding_uni = Embedding(input_dim=vocab_size, output_dim=300,
                              weights=[embedding_matrix], trainable=False,
                              input_length=max_len, mask_zero=True)(inputs_uni)

    embedding_bi = Embedding(input_dim=vocab_size, output_dim=300,
                             weights=[embedding_matrix], trainable=False,
                             input_length=max_len, mask_zero=True)(inputs_bi)

    # Concat embeddings of uni and bi to be fed to the model
    concatenated_embeddings = Concatenate()([embedding_uni, embedding_bi])

    # masking layer to allow the back propagation of masking the padding
    masking_layer = Masking()(concatenated_embeddings)

    # Bi-LSTM that will take inputs from masking layer and merge based on concat
    biLSTM = Bidirectional(LSTM(num_cells, dropout=dropout,
                                recurrent_dropout=rec_dropout,
                                return_sequences=True,
                                input_shape=(None, None, 300)),
                           merge_mode='concat')(masking_layer)

    # dense layer with softmax activation, with 4 classes
    out_dense = TimeDistributed(
        Dense(num_outputs, activation='softmax'))(biLSTM)

    # Defining model inputs and outputs
    model = Model([inputs_uni, inputs_bi], out_dense)

    # Defining Nesterov SGD optimizer with 0.95 momentum
    opt = SGD(lr=init_lr, momentum=momentum, nesterov=nesterov)

    # Compiling the model with temporal weights for the padding mask
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=None,
                  weighted_metrics=['accuracy'],
                  sample_weight_mode="temporal")

    # To visualize the model
    model.summary()

    # Plot the model to have an image for it (report purposes)
    plot_model(model, to_file='Bi-LSTM Unstacked model.png')

    logging.info("Bi-LSTM Unstacked model image saved")

    return model


def create_stacked_bilstm(config_params, vocab_dict):
    '''
    Creates an unstacked bi-lstm model with inputs shape of max_len
    argument passed,  it does use the loaded pretrained embeddings matrix,
    specifying num of cells as per LSTM, dropouts, and
    recurrent dropouts, finally, takes learning rate, momentum,
    nesterov params to build our optimizer

    Arguments:
        max_len
        vocab_size
        embedding_matrix
        num_cells
        dropout
        rec_dropout
        num_outputs
        init_lr
        momentum
        nesterov
    Returns:
        Model
    '''
    logging.info('Creating Bi-LSTM Stacked model')

    # load model params first from config params passed
    max_len = int(config_params['max_length'])
    num_cells = int(config_params['num_cells'])
    dropout = float(config_params['dropout'])
    rec_dropout = float(config_params['rec_dropout'])
    num_outputs = int(config_params['num_outputs'])
    init_lr = float(config_params['init_lr'])
    momentum = float(config_params['momentum'])
    nesterov = config_params['nesterov']

    vocab_size = len(vocab_dict)

    # load pretrained embeddings
    embedding_matrix = pretrained_embeddings(vocab_dict)

    # Create the unigrams & bigrams input layers
    inputs_uni = Input(shape=(max_len,))
    inputs_bi = Input(shape=(max_len,))

    # Create embeddings as per each layer
    # Not trainable because they are already pretrained
    # Output dim is 300, in order to have more contextual
    # and semantical meaning.
    # Of course masking zeros
    embedding_uni = Embedding(input_dim=vocab_size, output_dim=300,
                              weights=[embedding_matrix], trainable=False,
                              input_length=max_len, mask_zero=True)(inputs_uni)

    embedding_bi = Embedding(input_dim=vocab_size, output_dim=300,
                             weights=[embedding_matrix], trainable=False,
                             input_length=max_len, mask_zero=True)(inputs_bi)

    # Concat embeddings of uni and bi to be fed to the model
    concatenated_embeddings = Concatenate()([embedding_uni, embedding_bi])

    # masking layer to allow the back propagation of masking the padding
    masking_layer = Masking()(concatenated_embeddings)

    # Bi-LSTM that will take inputs from masking layer and merge based on sum
    bi_lstm_l1 = Bidirectional(LSTM(num_cells, dropout=dropout,
                                    recurrent_dropout=rec_dropout,
                                    return_sequences=True
                                    ),
                               merge_mode='sum')(masking_layer)

    # dense layer with softmax activation, with 4 classes
    out_dense_l1 = TimeDistributed(
        Dense(num_outputs, activation='softmax'))(bi_lstm_l1)

    # Stacking the model
    bi_lstm_l2 = Bidirectional(LSTM(num_cells, dropout=dropout,
                                    recurrent_dropout=rec_dropout,
                                    return_sequences=True
                                    ),
                               merge_mode='sum')(out_dense_l1)

    out_dense = TimeDistributed(
        Dense(num_outputs+1, activation='softmax'))(bi_lstm_l2)

    # Defining model inputs and outputs
    model = Model([inputs_uni, inputs_bi], out_dense)

    # Defining Nesterov SGD optimizer with 0.95 momentum
    opt = SGD(lr=init_lr, momentum=momentum, nesterov=nesterov)

    logging.info('Bi-LSTM Stacked model is created')

    # Compiling the model
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["acc"])

    # To visualize the model
    model.summary()

    # Plot the model to have an image for it (report purposes)
    plot_model(model, to_file='Bi-LSTM Stacked model.png')

    logging.info("Bi-LSTM Stacked model image saved")

    return model


def train_model(model, samples, train_y, epochs, batch_size, pad_mask):
    '''
    Trains the model we created earlier, shuffling both train_x and train_y,
    and split 20% of them for validation purposes, num of epochs, and,
    batch size are also passed
    Training callbacks are:
        1. tensorboard logger, in order to visualize training process
        2. Checkpointer: to save model after each epoch, not efficient
        3. Early Stopping: to avoid overfitting

    In case of any exception rising, we save the model and
    its weights before terminating.

    Arguments:
        model
        train_x
        train_y
        epochs
        batch_size
    Returns:
        history -- history of fitting our model to the data passed to it.
    '''
    logging.info("Starting training...")
    # to save the model
    model_path = os.path.join(
        os.getcwd(), 'models', 'model_weights.h5')

    # Log the model training process
    logger = TensorBoard("logging/pku_unstacked_keras_model")

    # Check pointer to save the model with best weights
    checkpointer = ModelCheckpoint(
        filepath=model_path, verbose=1, save_best_only=True,
        save_weights_only=True)

    # Early stopping, so it stops the model if it does not improve
    # Prevents overfitting
    early_stopping = EarlyStopping(
        monitor='loss', patience=5, verbose=1, mode='min')

    # Reducing Learning rate if stuck on a plateau
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=5, min_lr=0.0001)

    # Putting our callbacks in an array to be passed to the model
    cbk = [logger, checkpointer, early_stopping, reduce_lr]

    # For Uni & Bigrams
    # Unpacking our data for readability purposes
    train_x = samples[0]
    train_x_bi = samples[1]

    # Train the model with mask the padding
    # splitting training data for validation 80-20
    history = model.fit([train_x, train_x_bi], train_y,
                        epochs=epochs, batch_size=batch_size,
                        shuffle=True, sample_weight=pad_mask,
                        validation_split=0.2,
                        callbacks=cbk, verbose=1,
                        workers=0, use_multiprocessing=True)

    logging.info("Training complete.")
    return history


def eval_model(model, dev_x, dev_x_bi, batch_size):
    '''
    It takes the model, development samples and labels, and predict
    every batch. 
    Predict the labels and gets the argument max
    of the probabilites to assign as per class.

    :param model: model created or loaded
    :param dev_x: development set unigrams' samples
    :param dev_x_bi: development set bigrams' samples
    :param dev_y: development set labels
    :param batch_size

    :returns predictions
    '''
    # Use the trained model for predictions
    # Set it to run on main_thread and multiprocessing
    predictions = model.predict(
        x=[dev_x, dev_x_bi], batch_size=batch_size,
        verbose=1, workers=0,
        use_multiprocessing=True)

    # Get maximum of all probabilities
    predictions = np.argmax(predictions, axis=2)

    # Return the probabilities
    return predictions


if __name__ == "__main__":
    # Change logger printing format
    initialize_logger()

    # Read the config file
    config_file_path = os.path.join(os.getcwd(), 'config.yaml')
    config_file = open(config_file_path)
    config_params = yaml.load(config_file)

    # Load pretrained embeddings
    dict_json_file = os.path.join(os.getcwd(), 'vocab_dict.json')
    vocab_dict = json.loads(dict_json_file)
    embedding_matrix = pretrained_embeddings(vocab_dict)

    model = create_bilstm(config_params, vocab_dict)

    # For training from the last model saved
    # model_path = os.path.join(os.getcwd(), 'models', 'model-06.h5')
    # model = tf.keras.models.load_model(model_path)
    # logging.info(f'Model loaded from {model_path}')

    # model_path = os.path.join(os.getcwd(), 'model_weights.h5')
    # model.load_weights(model_path)

    epochs = config_params['epochs']
    batch_size = config_params['batch_size']

    try:
        samples = [train_x, train_x_bi]
        history = train_model(model, samples, train_y, epochs, batch_size)
    except KeyboardInterrupt:
        model_path = os.path.join(os.getcwd(), 'models', 'model.h5')
        model.save(model_path)

        model_weights_path = model_path.replace('model', 'model_weights')
        model.save_weights(model_weights_path)

    # logging.info("Evaluating test...")
    # loss_acc = model.evaluate(test_x, test_y, verbose=0)
    # logging.info(f"Test data: loss = {loss_acc[0]}, accuracy = {loss_acc[1]*100}% ")
