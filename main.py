import os
import argparse

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

import nsml
from nsml import DATASET_PATH

from dataset import Dataset
from tokenizer import Tokenizer

EP = 11
LR = 0.001
BATCH_SIZE = 128
MAX_SEQ = 32
VOCAB_SIZE = 2000
EMBEDDING_DIM = 128
epoch_idx = 1


def bind_model(model):
    def save(path, *args, **kwargs):
        # save the model with 'checkpoint' dictionary.
        model.model.save(os.path.join(path, 'model'))

    def load(path, *args, **kwargs):
        model = keras.models.load_model(os.path.join(path, 'model'))
        return model

    def infer(path, **kwargs):
        return inference(path, model, config)

    nsml.bind(save, load, infer)


def inference(path, model, config, **kwargs):
    test_dataset_path = DATASET_PATH+'/test'
    data = Dataset(
        test_dataset_path,
        "test_data",
        tokenizer,
        label_file_name=None,
        max_sequence_len=MAX_SEQ
    )
    
    pred_val = model.model.predict([data.input1, data.input2]).tolist()
    pred_results = [[step, val] for step, val in enumerate(pred_val)]
    return pred_results


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
        #y_pred = self.model.predict([self.x1, self.x2], verbose=0)
        #train_roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict([self.x1_val, self.x2_val], verbose=0)
        val_roc = roc_auc_score(self.y_val, y_pred_val)

        #train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        nsml.report(
            summary=True,
            step=epoch_idx,
            scope=locals(),
            **{
                #"train__epoch_score": float(train_roc),
                #"train__epoch_loss": float(train_loss),
                "valid__epoch_score": float(val_roc),
                "valid__epoch_loss": float(val_loss),
            })
        nsml.save(str(epoch_idx))
        epoch_idx += 1
        return


class Model():
    def __init__(self, lr):
        input1 = Input(shape=(MAX_SEQ,), )
        input2 = Input(shape=(MAX_SEQ,), )
        embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        emb1 = embedding_layer(input1)
        emb2 = embedding_layer(input2)
        x1 = LSTM(128, dropout=0.7, return_sequences=True)(emb1)
        #x1 = LSTM(256, dropout=0.7, return_sequences=True)(x1)
        x1 = LSTM(128, dropout=0.7)(x1)
        x2 = LSTM(128, dropout=0.7, return_sequences=True)(emb2)
        #x2 = LSTM(256, dropout=0.7, return_sequences=True)(x2)
        x2 = LSTM(128, dropout=0.7)(x2)
        feature = keras.layers.concatenate([x1,x2])
        #feature = Dense(256)(feature)
        #feature = BatchNormalization()(feature)
        #feature = Activation('relu')(feature)
        output = Dense(2, activation='sigmoid')(feature)
        self.model = keras.models.Model(inputs=[input1, input2], outputs=output)

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


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    
    config = args.parse_args()
    
    # Bind model
    model = Model(LR)
    bind_model(model)

    # Make tokenizer
    tokenizer = Tokenizer(VOCAB_SIZE, DATASET_PATH, config.mode)

    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == "train":
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

        my_callback = Roc_Auc_Callback(
            train_data.input1,
            train_data.input2,
            train_data.labels,
            valid_data.input1,
            valid_data.input2,
            valid_data.labels
        )
        #early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=6)
        model.model.fit(
            x=[train_data.input1, train_data.input2], 
            y=train_data.labels, 
            epochs=EP, 
            batch_size=BATCH_SIZE, 
            #callbacks=[my_callback, early_stopping],
            callbacks=[my_callback],
            validation_data=([valid_data.input1, valid_data.input2], valid_data.labels)
            )
        