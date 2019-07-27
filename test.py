import os
import argparse
from timeit import default_timer as timer
import random

import keras
import keras.models
from keras.callbacks import Callback, EarlyStopping
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.layers import Input, LSTM, Dense, Embedding, BatchNormalization, Dropout, Activation, Bidirectional
from keras.optimizers import RMSprop
import numpy as np
from sklearn.metrics import roc_auc_score


from dataset import Dataset
from tokenizer import Tokenizer

DATASET_PATH='./'
EP = 50
LR = 0.001
BATCH_SIZE = 128
MAX_SEQ = 64
VOCAB_SIZE = 100
EMBEDDING_DIM = 128
epoch_idx = 1

# Make tokenizer
tokenizer = Tokenizer(VOCAB_SIZE, DATASET_PATH)

class Roc_Auc_Callback(Callback):
    def __init__(self, trainX1, trainX2, trainY, validX1, validX2, validY):
        self.x1 = trainX1
        self.x2 = trainX2
        self.y = trainY
        self.x1_val = validX1
        self.x2_val = validX2
        self.y_val = validY
        self.best_epoch_idx = -1
        self.best_valid_score = 0
        self.early_stop_count = 0

    def on_epoch_end(self, epoch, logs={}):
        global epoch_idx
        y_pred = self.model.predict([self.x1, self.x2], verbose=0)
        train_roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict([self.x1_val, self.x2_val], verbose=0)
        val_roc = roc_auc_score(self.y_val, y_pred_val)

        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        """nsml.report(
            summary=True,
            step=epoch_idx,
            scope=locals(),
            **{
                "train__epoch_score": float(train_roc),
                "train__epoch_loss": float(train_loss),
                "valid__epoch_score": float(val_roc),
                "valid__epoch_loss": float(val_loss),
            })"""
        print({
                "train__epoch_score": float(train_roc),
                "train__epoch_loss": float(train_loss),
                "valid__epoch_score": float(val_roc),
                "valid__epoch_loss": float(val_loss),
            })
        epoch_idx += 1
        return


class Model():
    def __init__(self, lr):
        input1 = Input(shape=(MAX_SEQ,), )
        input2 = Input(shape=(MAX_SEQ,), )
        embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        emb1 = embedding_layer(input1)
        emb2 = embedding_layer(input2)
        x1 = LSTM(256, dropout=0.7)(emb1)
        x2 = LSTM(256, dropout=0.7)(emb2)
        feature = keras.layers.concatenate([x1,x2])
        #feature = Dense(256, activation='relu')(feature)
        output = Dense(2, activation='sigmoid')(feature)
        self.model = keras.models.Model(inputs=[input1, input2], outputs=output)
        self.model.summary()
        """self.model = Sequential([
            inputInput(shape=(5,), name='aux_input')
            Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQ),
            #Bidirectional(LSTM(128, dropout=0.7)),
            #LSTM(400, dropout=0.7, return_sequences=True),
            LSTM(256, dropout=0.7),
            
            #Dense(128, use_bias=False),
            #BatchNormalization(),
            #Activation('relu'),
            #Dropout(0.5),
            Dense(2, activation='sigmoid')
        ])"""

        rms_pr = RMSprop(lr=lr)
        self.model.compile(optimizer=rms_pr, loss='binary_crossentropy')

        return    



model = Model(LR)

# configs
train_dataset_path = DATASET_PATH + '/train/train_data'

# Load data
train_data = Dataset(
    train_dataset_path,
    'train_data',
    tokenizer,
    label_file_name='train_label',
    max_sequence_len=MAX_SEQ
    )
valid_data = Dataset(
    train_dataset_path, 
    'valid_data',
    tokenizer,
    label_file_name='valid_label',
    max_sequence_len=MAX_SEQ
    )

print(train_data.input1)
print(train_data.input2)

my_callback = Roc_Auc_Callback(
    train_data.input1,
    train_data.input2,
    train_data.labels,
    valid_data.input1,
    valid_data.input2,
    valid_data.labels
    )

early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=1)
model.model.fit(
    x=[train_data.input1, train_data.input2], 
    y=train_data.labels, 
    epochs=EP, 
    batch_size=BATCH_SIZE, 
    callbacks=[my_callback, early_stopping],
    validation_data=([valid_data.input1, valid_data.input2], valid_data.labels)
    )