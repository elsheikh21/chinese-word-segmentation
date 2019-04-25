import os
import logging
import tensorflow as tf
import json
from tqdm import tqdm


def initialize_logger():
    # For logging purposes
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def tensorflow_config():
    # Silence the logs of TF
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Allow growth of memory, so tensorflow can use whatever memory required
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)


def split_string(txt, limit, sep=" "):
    '''
    Takes a string and split it into multiple strings based on limit passed

    @param: txt -- str input
    @param: txt -- str input
    @param: txt -- str input

    @returns: list of texts
    '''
    texts = [txt[y-limit:y] for y in range(limit, len(txt)+limit, limit)]
    return texts


def save_dict_json(vocab_dict, file_name='vocab_dict.json'):
    '''
    Takes in a dictionary, and creates a json file with
    all of dictionary keys and values.

    Arguments:
        vocab_dict: {[str]} -- a path pointing to the file
    Returns:
        nothing, saves a json file in the same directory
    '''
    with open(file=file_name, encoding='utf-8', mode='w+') as json_file:
        json.dump(vocab_dict, json_file, ensure_ascii=False)
    logging.info(f'Dataset vocabulary is saved as "{file_name}"')


def concatenate_dataset(filenames, output_file='datasets_concatenated.utf8'):
    with open(output_file, encoding='utf-8', mode='w') as outfile:
        for fname in tqdm(filenames, desc='Concatenating datasets'):
            with open(fname, encoding='utf-8', mode='r') as infile:
                for line in tqdm(infile, desc='Processing dataset'):
                    outfile.write(line)
    logging.info(
        f'{len(filenames)} Datasets are concatenated to {output_file}')


if __name__ == "__main__":
    initialize_logger()

    parent_dir = os.getcwd()

    as_dataset = os.path.join(parent_dir, 'dataset',
                              'icwb2-data', 'training', 'as_training.utf8')
    cityu_dataset = os.path.join(
        parent_dir, 'dataset', 'icwb2-data', 'training', 'cityu_training.utf8')
    pku_dataset = os.path.join(
        parent_dir, 'dataset', 'icwb2-data', 'training', 'pku_training.utf8')
    msr_dataset = os.path.join(
        parent_dir, 'dataset', 'icwb2-data', 'training', 'msr_training.utf8')

    file_names = [cityu_dataset, pku_dataset, msr_dataset]
    concatenate_dataset(file_names, output_file='cityu_msr_pku_concat.utf8')
