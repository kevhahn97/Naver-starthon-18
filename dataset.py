import os

import nsml

from keras.utils import np_utils
from keras.preprocessing import sequence
import numpy as np

# tokenize, padding, flip, one-hot 전부 해야함
# tokenizer 받아와야 함

class Dataset():
    def __init__(
            self,
            data_dir,
            file_name,
            tokenizer,
            label_file_name=None,
            max_sequence_len=None
            
    ):
        self.data_file_name = file_name
        self.label_file_name = label_file_name
        self.data_dir = data_dir
        self.data_file_path = os.path.join(data_dir, file_name)
        if label_file_name:
            self.label_file_path = os.path.join(data_dir, label_file_name)
        else:
            self.label_file_path = None

        self.max_sequence_len = max_sequence_len
        self.tokenizer = tokenizer

        self._load_data(self.data_file_path, self.label_file_path)

    def _load_data(self, data_file_path, label_file_path):
        with open(data_file_path) as f:
            data = f.read().splitlines()
            data = [line.split("\t") for line in data]
            print(data)
            _, a_seqs, b_seqs = list(zip(*data))

            # texts
            a_ids = []
            b_ids = []
            print("preprocessing data")
            for a_seq, b_seq in zip(a_seqs, b_seqs):
                a_ids.append(self.tokenizer.text2ids(a_seq))
                b_ids.append(self.tokenizer.text2ids(b_seq))
            self.input1 = sequence.pad_sequences(a_ids, maxlen=self.max_sequence_len, truncating='post', padding='post')
            self.input2 = sequence.pad_sequences(b_ids, maxlen=self.max_sequence_len, truncating='post', padding='post')
            self.input1 = np.flip(self.input1, axis=1)
            self.input2 = np.flip(self.input2, axis=1)

        if label_file_path:
            with open(label_file_path) as f:
                labels = f.read().splitlines()
                labels = [int(label) for label in labels]

            self.labels = np_utils.to_categorical(labels)
        else:
            self.labels = None
