#!/usr/bin/env python
# encoding: utf-8
import os
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model

if __package__ != 'deepcut':
    from burmese_utils import CHAR_TYPE_FLATTEN, CHARS_MAP, CHAR_TYPES_MAP
    from utils import create_n_gram_df
    from model import get_convo_nn2
else:
    from .burmese_utils import CHAR_TYPE_FLATTEN, CHARS_MAP, CHAR_TYPES_MAP
    from .utils import create_n_gram_df
    from .model import get_convo_nn2


def generate_words(lines):
    """
    Iterate lines; remove sentence labels; return words.
    """

    words_all = []
    for line in lines:
        text = line.split('\t')[1].strip()
        words = text.split(' ')
        words_all.extend(words)
    return words_all


def create_char_dataframe(words):
    """
    Give list of input tokenized words,
    create dataframe of characters where first character of
    the word is tagged as 1, otherwise 0

    Example
    =======
    ['များ', 'သည်'] to dataframe of
    [{'char': 'မ', 'type': ..., 'target': 1}, ...,
     {'char': 'ျ', 'type': ..., 'target': 0}]
    """
    char_dict = []
    for word in words:
        for i, char in enumerate(word):
            if i == 0:
                char_dict.append({'char': char,
                                  'type': CHAR_TYPE_FLATTEN.get(char, 'o'),
                                  'target': True})
            else:
                char_dict.append({'char': char,
                                  'type': CHAR_TYPE_FLATTEN.get(char, 'o'),
                                  'target': False})
    return pd.DataFrame(char_dict)

def _ensure_dir_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def generate_alt_dataset(alt_path, output_path='cleaned_data', create_val=False):
    """
    Generate CSV file for training and testing data

    Input
    =====
    alt_path: str, path to ALT file which contains the raw corpus

    cleaned_data: str, path to output folder, the cleaned data will be saved
        in the given folder name where training set will be stored in `train` folder
        and testing set will be stored on `test` folder

    create_val: boolean, True or False, if True, divide training set into training set and
        validation set in `val` folder
    """

    _ensure_dir_exists(output_path)
    _ensure_dir_exists(os.path.join(output_path, 'train'))
    _ensure_dir_exists(os.path.join(output_path, 'test'))
    if create_val:
        _ensure_dir_exists(os.path.join(output_path, 'val'))

    with open(alt_path) as f:
        lines = f.read().splitlines()

    lines_train, lines_test = train_test_split(lines, random_state=0, test_size=0.1)
    if create_val:
        lines_train, lines_val = train_test_split(lines_train, random_state=0, test_size=0.1)
        val_words = generate_words(lines_val)
        val_df = create_char_dataframe(val_words)
        val_df.to_csv(os.path.join(output_path, 'val', 'df_alt_val.csv'), index=False)
    train_words = generate_words(lines_train)
    test_words = generate_words(lines_test)
    train_df = create_char_dataframe(train_words)
    test_df = create_char_dataframe(test_words)
    train_df.to_csv(os.path.join(output_path, 'train', 'df_alt_train.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test', 'df_alt_test.csv'), index=False)
    print("Saving to CSV file")


def prepare_feature(alt_processed_path, option='train'):
    """
    Transform processed path into feature matrix and output array

    Input
    =====
    alt_processed_path: str, path to processed BEST dataset

    option: str, 'train' or 'test'
    """
    # padding for training and testing set
    n_pad = 21
    n_pad_2 = int((n_pad - 1)/2)
    pad = [{'char': ' ', 'type': 'p', 'target': True}]
    df_pad = pd.DataFrame(pad * n_pad_2)

    df = []
    df.append(pd.read_csv(os.path.join(alt_processed_path, option, 'df_alt_{}.csv'.format(option))))
    df = pd.concat(df)
    df = pd.concat((df_pad, df, df_pad)) # pad with empty string feature

    df['char'] = df['char'].map(lambda x: CHARS_MAP.get(x, 80))
    df['type'] = df['type'].map(lambda x: CHAR_TYPES_MAP.get(x, 4))
    df_pad = create_n_gram_df(df, n_pad=n_pad)

    char_row = ['char' + str(i + 1) for i in range(n_pad_2)] + \
               ['char-' + str(i + 1) for i in range(n_pad_2)] + ['char']
    type_row = ['type' + str(i + 1) for i in range(n_pad_2)] + \
               ['type-' + str(i + 1) for i in range(n_pad_2)] + ['type']

    x_char = df_pad[char_row].as_matrix()
    x_type = df_pad[type_row].as_matrix()
    y = df_pad['target'].astype(int).as_matrix()

    return x_char, x_type, y


def train_model(alt_processed_path, weight_path='../weight/model_weight.h5', verbose=2):
    """
    Given path to processed ALT dataset,
    train CNN model for words beginning alongside with
    character label encoder and character type label encoder

    Input
    =====
    alt_processed_path: str, path to processed ALT dataset
    weight_path: str, path to weight path file
    verbose: int, verbost option for training Keras model

    Output
    ======
    model: keras model, keras model for tokenize prediction
    """

    x_train_char, x_train_type, y_train = prepare_feature(alt_processed_path, option='train')
    x_test_char, x_test_type, y_test = prepare_feature(alt_processed_path, option='test')

    validation_set = False
    if os.path.isdir(os.path.join(alt_processed_path, 'val')):
        validation_set = True
        x_val_char, x_val_type, y_val = prepare_feature(alt_processed_path, option='val')

    _ensure_dir_exists(os.path.dirname(weight_path))

    # TODO: never saves the model!
    callbacks_list = [
        ModelCheckpoint(
            weight_path,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            verbose=verbose
        )
    ]

    # train model
    # TODO: why is +1 required here?
    model = get_convo_nn2(no_char=len(CHARS_MAP.keys())+1, no_type=len(CHAR_TYPES_MAP.keys()))
    plot_model(model, to_file="model.svg", show_shapes=True)
    train_params = [(10, 256), (3, 512), (3, 2048), (3, 4096), (3, 8192)]
    for (epochs, batch_size) in train_params:
        print("train with {} epochs and {} batch size".format(epochs, batch_size))
        if validation_set:
            model.fit([x_train_char, x_train_type], y_train,
                      epochs=epochs, batch_size=batch_size,
                      verbose=verbose,
                      callbacks=callbacks_list,
                      validation_data=([x_val_char, x_val_type], y_val))
        else:
            model.fit([x_train_char, x_train_type], y_train,
                      epochs=epochs, batch_size=batch_size,
                      verbose=verbose,
                      callbacks=callbacks_list)
    return model


def evaluate(alt_processed_path, model):
    """
    Evaluate model on splitted 10 percent testing set
    """
    x_test_char, x_test_type, y_test = prepare_feature(alt_processed_path, option='test')

    y_predict = model.predict([x_test_char, x_test_type])
    y_predict = (y_predict.ravel() > 0.5).astype(int)

    f1score = f1_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)

    return f1score, precision, recall
