import logging
import os
from pathlib import Path

from tqdm import tqdm
from utils import initialize_logger


def text_bies_format(text, separator=''):
    '''
    Takes in a string, split on whitespace, generating an array
    loop on the array

   Arguments:
        text: str - - Input maybe have whitespaces, numbers, punctuation
        separator: str - - Set by default to empty join
    '''
    # Split input on whitespace
    split_text = text.split()
    # Written in a separate line for readability purposes
    split_len = len(split_text)
    # Creating an array to have
    labels = []
    # Loop on words of array of text we have
    for i in range(split_len):
        # To add label per word
        lbl = []
        #  for readability purposes
        txt = split_text[i]
        # length of the word
        txt_len = len(txt)
        # Punctuation symbols are of len = 1
        if(txt_len == 1):
            labels.append('S')
            continue
        else:
            # otherwise words will be trimmed first and last
            txt = txt[1: -1]
            # Since we trimmed the first char in word
            lbl.append('B')
            for _ in range(len(txt)):
                lbl.append('I')
            # Since we trimmed the last char in word
            lbl.append('E')
            # joining the list of label characters
            lbl = separator.join(lbl)
            labels.append(lbl)
    # Join the labels to be returned in text format
    text_label = separator.join(labels)
    return text_label


def bies_format(file_path):
    '''
    Takes file path, reads from it text and invokes text_bies_format
    to create BIES labels for each text. Writes the BIES labels in
    labels_file.txt file in same directory as the file_path parameter,
    and produce input_file.txt saved in same directory, as per the
    file_path passed to it.


    Arguments:
        file_name: str -- points to file you need to get its BIES labels
    '''
    # Check if file is not there
    if not (os.path.isfile(file_path)):
        # Print error message for user, without raising exceptions.
        print('[ERROR] File does not exist')
    else:
        # to add labels of the file
        file_labels_lst, file_input_lst, file_content = [], [], None
        with open(file_path, encoding="utf-8", mode="r") as file:
            # Reading the file
            file_content = file.readlines()

            length = len(file_content)

            for i in tqdm(range(length), desc='Processing original file...'):
                # readability purposes
                txt = file_content[i]
                # appending the line after splitting it to input list
                txt_no_whitespace = txt.split()
                txt_no_whitespace = ''.join(txt_no_whitespace)
                file_input_lst.append(txt_no_whitespace)
                # appending the line after getting its labels to labels list
                file_labels_lst.append(text_bies_format(txt))

        # Getting the parent path of the file path given
        parent_path = Path(file_path).parent
        # Creating a labels file, if it does not exist; to save the output in
        new_files_name = file_path.replace('.utf8', '')
        lbl_file_name = f'{new_files_name}_labels_file.txt'
        label_file = os.path.join(parent_path, lbl_file_name)
        tmp_label_file = Path(label_file)
        tmp_label_file.touch(exist_ok=True)
        # Creating an input file, replicating what was done for labels file
        input_file_name = f'{new_files_name}_input_file.txt'
        input_file = os.path.join(parent_path, input_file_name)
        tmp_input_file = Path(input_file)
        tmp_input_file.touch(exist_ok=True)

        # Opening the created labels file to save the labels in it
        with open(label_file, mode="w+") as labels_file:
            if not (len(file_labels_lst) == 0):
                for i in tqdm(range(len(file_labels_lst)),
                              desc='Creating the labels file...'):
                    labels_file.write(file_labels_lst[i] + '\n')
        logging.info(f'Labels file created and saved at "{label_file}"')

        # Opening the created input file to save the input in it
        with open(input_file, encoding='utf-8', mode="w+") as inputs_file:
            if not (len(file_input_lst) == 0):
                for i in tqdm(range(len(file_input_lst)),
                              desc='Creating the Inputs file...'):
                    inputs_file.write(file_input_lst[i] + '\n')
        logging.info(f'Inputs file created and saved at "{input_file}"')
        return input_file_name, lbl_file_name


def generate_labels_dict():
    '''
    We have 4 labels B, I , E, S. Each representing Beginning, Inside,
    End, Single.
    We produce a hot encoded array of them and build a dictionary
    to map those numbers to labels back and forth

    Returns:
        encoded_dict: {[dict]} -- a dictionary contains keys as labels,
                                    and values as label hot encoding val
        rev_encoded_dict: {[dict]} -- encoded_dict but reversed
    '''
    Y = ['B', 'I', 'E', 'S']
    encoded_dict, rev_encoded_dict = dict(), dict()
    for i, y in enumerate(Y):
        encoded_dict[y] = i
        rev_encoded_dict[i] = y
    return encoded_dict, rev_encoded_dict


if __name__ == "__main__":
    initialize_logger()

    dataset_name = 'as_training.utf8'
    training_file = os.path.join(os.getcwd(), dataset_name)
    logging.info(f'Working on dataset in {training_file}')

    input_file_name, labels_file_name = bies_format(training_file)

    encoded_dict, rev_encoded_dict = generate_labels_dict()
