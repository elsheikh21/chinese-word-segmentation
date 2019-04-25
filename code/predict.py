import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dataset_handling import generate_labels_dict, process_dev_dataset
from model import eval_model
from preprocessing import bies_format
import logging
from utils import initialize_logger

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path",
                        help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the
    checkpoint and write a new file (output_path)
    with your predictions in the BIES format.

    The resources folder should contain everything you need
    to make the predictions.
    It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead,
    otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file
                        (where you save your predictions)
    :param resources_path: the path of the resources folder containing your
                            model and stuff you might need.
    :return: None
    """
    # TODO: REMOVE LABELS WHEN
    input_file_name, labels_file_name = bies_format(input_path)

    # Load config.yaml file
    config_file_path = os.path.join(resources_path, 'config.yaml')
    config_file = open(config_file_path)
    config_params = yaml.load(config_file)

    # Get vocabulary dictionary -saved as json file- path
    vocab_dict_json = os.path.join(
        resources_path, 'vocab_dict.json')

    # Open and read the file content
    input_content = []
    with open(input_file_name, encoding='utf-8', mode='r') as input_file:
        input_content = input_file.read().splitlines()

    # To save every line original length
    original_line_length = []
    for line in input_content:
        original_line_length.append(len(line))

    # Setting max_length as per the max in our input file
    max_len = max(original_line_length)

    # Read the development set, and pad it as per max line
    x_lst, x_lst_bi = process_dev_dataset(input_file_name, vocab_dict_json)

    dev_x = pad_sequences(np.array(x_lst), truncating='pre',
                          padding='post', maxlen=max_len, value=0)
    dev_x_bi = pad_sequences(np.array(x_lst_bi), truncating='pre',
                             padding='post', maxlen=max_len, value=0)

    logging.info('Data is padded and ready for model eval')

    # get the labels dict
    _, rev_encoded_dict = generate_labels_dict()

    # Get batch size from config file
    batch_size = config_params['batch_size']

    # Load the model
    logging.info('Loading model.')
    model_path = os.path.join(resources_path, 'model.h5')
    model = tf.keras.models.load_model(model_path)
    logging.info('Model is loaded.')

    logging.info('Evaluating the model')
    # Get the model prediction
    predictions = eval_model(model, dev_x, dev_x_bi, batch_size)

    logging.info('Writing down the predictions in output file')
    # Convert map our predictions to their labels
    # remove padding from them
    predicted_classes = []
    for (prediction, line_len) in tqdm(zip(predictions, original_line_length),
                                       desc='encoding predictions'):
        prediction_ = prediction[:line_len]
        encoded_prediction = [rev_encoded_dict.get(number)
                              for number in prediction_]
        predicted_classes.append(encoded_prediction)

    # Write our predictions to the output file
    with open(output_path, mode='w+') as file:
        for i in tqdm(range(len(predicted_classes)),
                      desc='Writing model predictions'):
            file.write(''.join(predicted_classes[i]) + '\n')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    initialize_logger()
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
    # For testing purposes
    # input_path = 'C:/Users/Sheikh/Documents/GitLab/chinese-word-segmentation/dataset/icwb2-data\development/cityu_test_gold.utf8'
    # output_path = 'C:/Users/Sheikh/Documents/GitLab/chinese-word-segmentation/code/model_predictions.txt'
    # resources_path = 'C:/Users/Sheikh/Documents/GitLab/chinese-word-segmentation/resources'
    # predict(input_path, output_path, resources_path)


# python -u predict.py C:\Users\Sheikh\Documents\GitLab\chinese-word-segmentation\dataset\icwb2-data\development\as_testing_gold.utf8 C:\Users\Sheikh\Documents\GitLab\chinese-word-segmentation\output.txt C:\Users\Sheikh\Documents\GitLab\chinese-word-segmentation\resources
