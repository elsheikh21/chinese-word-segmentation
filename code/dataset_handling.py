import json
import logging
import os

import numpy as np
from keras.utils import to_categorical
from nltk import bigrams
from nltk.util import ngrams
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from utils import (initialize_logger, save_dict_json, split_string,
                   tensorflow_config)

from preprocessing import bies_format, generate_labels_dict


def simplify_dataset(input_file):
    '''
    Convert dataset from Traditional to Simplified Chinese.
    Appends "simplified_" to the input file name and saves it
    in current directory

    Arguments:
        input_file: str -- string points to the path of the dataset
    '''
    if('.utf8' not in input_file):
        print(f"[Warning] the file has to be of ‘utf8' format.")
    else:
        output_file = f'simplified_{input_file}'
        cmd = f'hanzi-convert -o {output_file} -s {input_file}'
        os.system(cmd)
        print(f"[ INFO] Simplified dataset is saved in {output_file}")


def get_dataset_chars(file_name):
    '''
    This method takes in the dataset file and returns an array of
    all the words in the dataset

    Arguments:
        file_name -- expecting a file with utf-8 encoding

    Returns:
        unique_chars -- an array containing all unique characters
                        in the dataset
        char_to_idx -- dictionary containing all unigrams and their indices
        idx_to_char -- rev dictionary of char_to_idx
    '''
    raw_dataset, unique_chars, char_to_idx = None, None, dict()
    with open(file_name, encoding='utf-8', mode='r') as f:
        lines = f.read().splitlines()
        raw_dataset = ' '.join(' '.join(lines).split())
        unique_chars = set(raw_dataset)
    char_to_idx["<PAD>"] = 0
    char_to_idx["<UNK>"] = 1
    char_to_idx.update(dict([(char, i)
                             for i, char in enumerate(unique_chars, start=1)]))
    idx_to_char = dict(zip(char_to_idx.values(), char_to_idx.keys()))
    unique_chars_list = list(unique_chars)

    return unique_chars_list, char_to_idx, idx_to_char


def get_dataset_bigrams(file_name):
    '''
    Reads the file, from the given path and get its content.
    using nltk.utils.ngrams to get the bigrams from the file_content
    typecasting them to list and creating an iteratables of 2 elements
    to create a joint of 2 chars in a list

    Arguments:
        file_name: {[str]} -- points to the path where the file is located
    Return:
        list: {[list]} -- a list of unique bigrams in the given dataset
        bi_grams_dict: {[dictionary]} -- a dictionary as per bigrams only
        rev_bi_grams_dict: {[dictionary]} -- bi_grams_dict but reversed
    '''
    logging.info(f'Reading contents of {file_name}')
    with open(file_name, encoding='utf-8', mode='r') as f:
        file_content = f.read().splitlines()

    logging.info('Creating bigrams from it...')
    bigrams_list = []
    for i in (range(len(file_content))):
        # Removing whitespaces if there are any, defensive paradigm
        txt = file_content[i].replace(' ', '')
        # Creating the n grams as per text
        bigram = ngrams(txt, 2)
        # Add bigrams after typecasting to lists
        bigrams_list.append(list(bigram))

    # zip to create an iteratable, and then join tuples to create your bigrams
    ngrams_ = zip(*[bigrams_list[i:] for i in range(2)])
    bigrams_list = []
    # Looping over the ngrams iteratable object created
    for _, ngram in tqdm(ngrams_, desc='Generating Bigrams'):
        # looping over the bigrams mini lists
        for gram in ngram:
            # Joining list elements to have bigrams as strings
            bigrams_list.append(''.join(gram))
    # Getting all unique bigrams
    unique_bigrams = set(bigrams_list)
    # Creating a list of them, so to create a dictionary
    unique_bigrams_list = list(unique_bigrams)

    # Creating dict and reversed dict as per list
    bigrams_to_idx = {val: key for (
        key, val) in enumerate(unique_bigrams_list)}
    idx_to_bigrams = {val: key for (key, val) in bigrams_to_idx.items()}

    # Returning everything created
    logging.info('Bigrams are created.')
    return unique_bigrams_list, bigrams_to_idx, idx_to_bigrams


def create_vocab_dict(unigrams_list, bigrams_list):
    '''
    Takes in the unigrams_list and bigrams list,
    creates a new list having both elements. And
    from this list creates a dictionary

    Arguments:
        unigrams_list: {[list]} -- list containing all unique unigrams
        bigrams_list: {[list]} -- list containing all unique bigrams

    Returns:
        vocab_dict: {[dict]} -- dictionary has <pad> and <unk> keys
        as well as, all the unigrams and bigrams each has unique value

        rev_vocab_dict: {[dict]} -- same as vocab_dict but keys and
        values are interchanged
    '''
    unigrams_list.extend(bigrams_list)
    vocab_dict = dict()
    vocab_dict['<PAD>'] = 0
    vocab_dict['<UNK>'] = 1
    vocab_dict.update(
        {val: key for (key, val) in enumerate(unigrams_list, start=2)})
    rev_vocab_dict = {val: key for (key, val) in vocab_dict.items()}
    return unigrams_list, vocab_dict, rev_vocab_dict


def get_labels(labels_file_name, max_length, encoded_dict):
    labels_file_content, y_lst = [], []
    # Read in the labels file
    with open(labels_file_name, mode='r') as file:
        labels_file_content = file.read().splitlines()

    # Use encoded_dict to get the equivalent one hot encoding as per label
    # as per line of the file
    for i in tqdm(range(len(labels_file_content)),
                  desc='Reading labels file...'):
        txt = labels_file_content[i].strip()
        if(len(txt) > max_length):
            lines = split_string(txt, max_length)
            for txt in lines:
                txt = [encoded_dict[label] for label in txt]
                y_lst.append(txt)
        else:
            txt = [encoded_dict[label] for label in txt]
            y_lst.append(txt)

    return y_lst


def get_unigrams(input_file_name, max_length, vocab_dict):
    input_file_content, x_lst = [], []
    # Read in the inputs file
    with open(input_file_name, encoding='utf-8', mode='r') as file:
        input_file_content = file.read().splitlines()

    for i in tqdm(range(len(input_file_content)),
                  desc='Constructing unigrams inputs...'):
        txt = input_file_content[i].strip()
        if(len(txt) > max_length):
            lines = split_string(txt, max_length)
            for txt in lines:
                txt = [vocab_dict.get(char, 1) for char in txt]
                x_lst.append(txt)
        else:
            txt = [vocab_dict.get(char, 1) for char in txt]
            x_lst.append(txt)
    return x_lst


def get_bigrams(input_file_name, max_length, vocab_dict):
    input_file_content, x_lst_bi = [], []
    # Read in the inputs file
    with open(input_file_name, encoding='utf-8', mode='r') as file:
        input_file_content = file.read().splitlines()

    # Use vocab_dict to get the equivalent idx as per char
    # as per line of the file
    for i in tqdm(range(len(input_file_content)),
                  desc='Constructing bigrams inputs...'):
        txt = input_file_content[i].strip()
        if(len(txt) > max_length):
            lines = split_string(txt, max_length)
            for txt in lines:
                bigram = list(bigrams(txt))
                txt_bigrams = list(map(''.join, bigram))
                txt = [vocab_dict.get(bigram, 1) for bigram in txt_bigrams]
                x_lst_bi.append(txt)
        else:
            bigram = list(bigrams(txt))
            txt_bigrams = list(map(''.join, bigram))
            txt = [vocab_dict.get(bigram, 1) for bigram in txt_bigrams]
            x_lst_bi.append(txt)
    return x_lst_bi


def read_dataset(input_file_name, labels_file_name, max_length):
    '''
    Reads the dataset, labels, pad the input to fit the max_length, after
    that it splits data into training and dev sets

    Arguments:
        input_file_name
        labels_file_name
        max_length: used to have a number of characters for the model

    Returns:
        train_x: numpy array of the character indices padded for max_len
        train_y:  numpy array of the character labels padded for max_len
    '''
    # Get unique chars list from dataset
    unique_chars_list, _, _ = get_dataset_chars(input_file_name)
    # Get unique bigrams list from dataset
    unique_bigrams_list, _, _ = get_dataset_bigrams(input_file_name)

    # Construct vocabulary
    _, vocab_dict, _ = create_vocab_dict(
        sorted(unique_chars_list), sorted(unique_bigrams_list))

    # save vocab dictionary to json file for later use
    save_dict_json(vocab_dict)

    logging.info('Dataset Reading in process...')

    (x_lst, x_lst_bi, y_lst) = [], [], []

    # Get the one hot encoding as per label
    encoded_dict, _ = generate_labels_dict()

    logging.info('Labels Generated.')

    y_lst = get_labels(labels_file_name, max_length, encoded_dict)

    x_lst = get_unigrams(input_file_name, max_length, vocab_dict)

    x_lst_bi = get_bigrams(input_file_name, max_length, vocab_dict)

    logging.info(f'Padding inputs, labels to {max_length} tokens')

    # When truncating, get rid of initial words and keep last part of the text
    # When padding, pad at the end of the sentence. (shorter sentences)
    train_x = pad_sequences(np.array(x_lst), truncating='pre',
                            padding='post', maxlen=max_length, value=0)
    train_x_bi = pad_sequences(np.array(
        x_lst_bi), truncating='pre', padding='post',
        maxlen=max_length, value=0)
    train_y = pad_sequences(np.array(y_lst), truncating='pre',
                            padding='post', maxlen=max_length, value=0)

    # categorize outputs
    train_y = to_categorical(train_y)

    logging.info('Dataset preparation process is done')

    return train_x, train_x_bi, train_y, vocab_dict


def process_dataset(input_file_name, labels_file_name):
    '''
    Reads the dataset, labels, pad the input to fit the max_length, after
    that it splits data into training and dev sets

    Arguments:
        input_file_name
        labels_file_name
        max_length: used to have a number of characters for the model

    Returns:
        train_x: numpy array of the character indices padded for max_len
        train_y:  numpy array of the character labels padded for max_len
    '''
    # Get unique chars list from dataset
    unique_chars_list, _, _ = get_dataset_chars(input_file_name)
    # Get unique bigrams list from dataset
    unique_bigrams_list, _, _ = get_dataset_bigrams(input_file_name)

    # Construct vocabulary
    _, vocab_dict, _ = create_vocab_dict(
        sorted(unique_chars_list), sorted(unique_bigrams_list))

    # save vocab dictionary to json file for later use
    save_dict_json(vocab_dict)

    logging.info('Dataset Reading in process...')

    (x_lst, x_lst_bi, y_lst) = [], [], []

    # Get the one hot encoding as per label
    encoded_dict, _ = generate_labels_dict()

    logging.info('Labels Generated.')

    y_lst = get_labels(labels_file_name, 4000, encoded_dict)

    x_lst = get_unigrams(input_file_name, 4000, vocab_dict)

    x_lst_bi = get_bigrams(input_file_name, 4000, vocab_dict)

    train_x = (x_lst)
    train_x_bi = (x_lst_bi)
    train_y = (y_lst)

    logging.info('Dataset preparation process is done')

    return train_x, train_x_bi, train_y, vocab_dict


def pad_per_batch(x, x_bi, y, batch_size, shuffle=False):
    '''
    Generates batches of 'batch_size' from the lists x, x_bi, y.
    And user decides to shuffle or not.
    Lists are padded as per maximum samples length as per batch.

    :param x
    :param y
    :param x_bi
    :param batch_size

    :yields a generator of batched size of the lists
    '''
    if not shuffle:
        for start in range(0, len(x), batch_size):
            end = start + batch_size

            x_, x_bi_, y_ = x[start:end], x_bi[start:end], y[start:end]

            max_len = len(max(x_, key=len))

            x_ = pad_sequences(np.array(x_), truncating='pre',
                               padding='post', maxlen=max_len, value=0)
            x_bi_ = pad_sequences(np.array(x_bi_), truncating='pre',
                                  padding='post', maxlen=max_len, value=0)
            y_ = pad_sequences(np.array(y_), truncating='pre',
                               padding='post', maxlen=max_len, value=0)

            # categorize outputs
            y_ = to_categorical(y_)

            yield x_, x_bi_, y_
    else:
        perm = np.random.permutation(len(x))
        for start in range(0, len(x), batch_size):
            end = start + batch_size

            x_, x_bi_ = x[perm[start:end]], x_bi[perm[start:end]]
            y_ = y[perm[start:end]]

            x_ = pad_sequences(np.array(x_), truncating='pre',
                               padding='post', maxlen=max_len, value=0)
            x_bi_ = pad_sequences(np.array(x_bi_), truncating='pre',
                                  padding='post', maxlen=max_len, value=0)
            y_ = pad_sequences(np.array(y_), truncating='pre',
                               padding='post', maxlen=max_len, value=0)

            # categorize outputs
            y_ = to_categorical(y_)

            yield x_, x_bi_, y_


def process_dev_dataset(input_file_name, vocab_dict_json):
    '''
    Reads the dataset, labels, pad the input to fit the max_length, after
    that it splits data into training and dev sets

    Arguments:
        input_file_name
        labels_file_name
        max_length: used to have a number of characters for the model

    Returns:
        train_x: numpy array of the character indices padded for max_len
        train_y:  numpy array of the character labels padded for max_len
    '''
    with open(vocab_dict_json, encoding='utf-8', mode='r') as f:
        vocab_dict = json.load(f)

    logging.info('Dataset Reading in process...')

    (x_lst, x_lst_bi) = [], []

    logging.info('Labels Generated.')

    x_lst = get_unigrams(input_file_name, 4000, vocab_dict)

    x_lst_bi = get_bigrams(input_file_name, 4000, vocab_dict)

    logging.info('Dataset preparation process is done')

    return x_lst, x_lst_bi,


if __name__ == "__main__":
    initialize_logger()

    # Simplifying all traditional datasets
    dataset_names = ['as_training.utf8', 'cityu_training.utf8']
    for dataset in dataset_names:
        dataset = os.path.join(os.getcwd(), dataset)
        simplify_dataset(dataset)

    # Getting BIES format
    dataset_name = 'as_training.utf8'
    training_file = os.path.join(os.getcwd(), dataset_name)
    logging.info(f'Working on dataset in {training_file}')

    input_file_name, labels_file_name = bies_format(training_file)

    # Get dataset unigram chars
    (unique_chars_list, char_to_idx, _) = get_dataset_chars(training_file)
    logging.info(f'Unigrams list is created, its length is {len(char_to_idx)}')

    # Get dataset bigram chars
    unique_bigrams_list, bigrams_to_idx, _ = get_dataset_bigrams(
        input_file_name)

    # create vocab dictionary based on unigrams and bigrams
    unigrams_bigrams_list, vocab_dict, _ = create_vocab_dict(
        unique_chars_list, unique_bigrams_list)
    vocab_size = len(vocab_dict)

    logging.info(f'Vocabulary created using  {vocab_size} unique ngram chars')

    # Save our vocabulary in a json file for later use
    save_dict_json(vocab_dict)

    # prepare our dataset
    MAX_LENGTH = 70

    input_file_name.replace(os.getcwd(), '').replace('\\', '')
    labels_file_name.replace(os.getcwd(), '').replace('\\', '')

    input_file_dir = os.path.join(os.getcwd(), input_file_name)
    labels_file_dir = os.path.join(os.getcwd(), labels_file_name)

    train_x, train_x_bi, train_y = read_dataset(
        input_file_dir, labels_file_dir, MAX_LENGTH, vocab_dict)

    logging.info(f'Training: input: {train_x.shape}, output: {train_y.shape}')
