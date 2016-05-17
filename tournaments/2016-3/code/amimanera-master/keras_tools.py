#!/usr/bin/python 
# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation, Reshape, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.embeddings import Embedding
from keras.utils import np_utils, generic_utils
from sklearn.base import BaseEstimator
import theano.tensor as T

'''
	This demonstrates how to reach a score of 0.4890 (local validation)
	on the Kaggle Otto challenge, with a deep net using Keras.
	Compatible Python 2.7-3.4
	Recommended to run on GPU:
		Command: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python kaggle_otto_nn.py
		On EC2 g2.2xlarge instance: 19s/epoch. 6-7 minutes total training time.
	Best validation score at epoch 21: 0.4881
	Try it at home:
		- with/without BatchNormalization (BatchNormalization helps!)
		- with ReLU or with PReLU (PReLU helps!)
		- with smaller layers, largers layers
		- with more layers, less layers
		- with different optimizers (SGD+momentum+decay is probably better than Adam!)
	Get the data from Kaggle: https://www.kaggle.com/c/otto-group-product-classification-challenge/data
'''

'''
From kaggle forum:

NN is the average of 30 neural networks with the same parameters fed by x^(2/3) transformed features and by results of KNN with N = 27 (KNN gained .002 for my best solution). NN was implemented on Keras, I've found this library very nice and fast (with CUDA-enabled Theano). Layers were (512,256,128), the score was .428

Dropout(.15) -> Dense(n_in, l1, activation='tanh') -> BatchNormalization((l1,)) -> Dropout(.5) -> Dense(l1, l2) -> PReLU((l2,)) -> BatchNormalization((l2,)) -> Dropout(.3) -> Dense(l2, l3) -> PReLU((l3,)) -> BatchNormalization((l3,)) -> Dropout(.1) -> Dense(l3, n_out) -> Activation('softmax')
sgd = SGD(lr=0.004, decay=1e-7, momentum=0.99, nesterov=True)


Rossmann 3d place: https://github.com/entron/category-embedding-rossmann/blob/master/models.py "categorical embedding"

'''


def RMSPE(y_true, y_pred):
    # y_true = T.exp(y_true)
    # y_pred = T.exp(y_pred)
    loss = T.sqrt(T.sqr((y_true - y_pred) / y_true).mean(axis=-1))
    return loss


def RMSE(y_true, y_pred):
    loss = T.sqrt(T.sqr(y_true - y_pred).mean(axis=-1))
    return loss


class KerasNN_OLD(Sequential, BaseEstimator):
    def __init__(self, dims=93, nb_classes=9, nb_epoch=50, learning_rate=0.004, validation_split=0.0, batch_size=128,
                 verbose=1):
        Sequential.__init__(self)
        self.dims = dims
        self.nb_classes = nb_classes
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.verbose = verbose
        print('Initializing Keras Deep Net with %d features and %d classes' % (self.dims, self.nb_classes))

        self.add(Dropout(0.15))
        self.add(Dense(dims, 512, activation='tanh'))
        self.add(BatchNormalization((512,)))
        self.add(Dropout(0.5))

        self.add(Dense(512, 256))
        self.add(PReLU((256,)))
        self.add(BatchNormalization((256,)))
        self.add(Dropout(0.3))

        self.add(Dense(256, 128))
        self.add(PReLU((128,)))
        self.add(BatchNormalization((128,)))
        self.add(Dropout(0.1))

        self.add(Dense(128, nb_classes))
        self.add(Activation('softmax'))

        sgd = SGD(lr=self.learning_rate, decay=1e-7, momentum=0.99, nesterov=True)
        self.compile(loss='categorical_crossentropy', optimizer=sgd)

    def fit(self, X, y):
        y = np_utils.to_categorical(y)
        Sequential.fit(self, X, y, nb_epoch=self.nb_epoch, batch_size=self.batch_size,
                       validation_split=self.validation_split)

    def predict_proba(self, Xtest):
        # ypred = Sequential.predict_proba(self,Xtest,batch_size=128,verbose=1)
        ypred = Sequential.predict_proba(self, Xtest, batch_size=self.batch_size, verbose=self.verbose)
        print(ypred.shape)
        return ypred

class KerasNN(BaseEstimator):
    def __init__(self, dims=66, nb_classes=1, nb_epoch=30, learning_rate=0.5, validation_split=0.0, batch_size=64,
                 loss='categorical_crossentropy', layers=[32,32], dropout=[0.2,0.2],verbose=1):

        self.dims = dims
        self.nb_classes = nb_classes
        self.classes_ = None # list containing classes
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.loss = loss
        self.layers = layers
        self.dropout = dropout
        self.verbose = verbose

        self.model = Sequential()
        # Keras model
        for layer,dropout in zip(self.layers,self.dropout):
            self.model.add(Dense(output_dim=layer, input_dim=dims, init='lecun_uniform'))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(dropout))


        self.model.add(Dense(output_dim=nb_classes))
        self.model.add(Activation('softmax'))

        #sgd = SGD(lr=self.learning_rate, decay=1e-7, momentum=0.99, nesterov=True)
        print('Compiling Keras Deep Net with loss: %s' % (str(loss)))
        self.model.compile(loss=loss, optimizer="adadelta")

    def fit(self, X, y, sample_weight=None):
        print('Fitting  Keras Deep Net for regression with batch_size %d, epochs %d  and learning rate: %f' % (
        self.batch_size, self.nb_epoch, self.learning_rate))
        y = np_utils.to_categorical(y)
        self.classes_ = np.unique(y)
        self.model.fit(X, y, nb_epoch=self.nb_epoch, batch_size=self.batch_size,
                       validation_split=self.validation_split)

    def predict_proba(self, Xtest):
        ypred = self.model.predict_proba(Xtest, batch_size=self.batch_size, verbose=self.verbose)
        return ypred

    def predict(self, Xtest):
        ypred = self.model.predict(Xtest, batch_size=self.batch_size, verbose=self.verbose)
        return np_utils.probas_to_classes(ypred)

class KerasEnsembler(BaseEstimator): # AUC 0.52558
    def __init__(self, dims=66, nb_classes=1, nb_epoch=30, learning_rate=0.5, validation_split=0.0, batch_size=64,
                 loss='categorical_crossentropy', verbose=1):

        self.dims = dims
        self.nb_classes = nb_classes
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.loss = loss
        self.verbose = verbose

        self.model = Sequential()
        # Keras model
        self.model.add(Dense(output_dim=16, input_dim=dims, init='lecun_uniform'))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Dense(output_dim=16, init='lecun_uniform'))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Dense(output_dim=nb_classes))
        self.model.add(Activation('softmax'))

        #sgd = SGD(lr=self.learning_rate, decay=1e-7, momentum=0.99, nesterov=True)
        print('Compiling Keras Deep Net with loss: %s' % (str(loss)))
        self.model.compile(loss=loss, optimizer="adadelta")

    def fit(self, X, y, sample_weight=None):
        print('Fitting  Keras Deep Net for regression with batch_size %d, epochs %d  and learning rate: %f' % (
        self.batch_size, self.nb_epoch, self.learning_rate))
        y = np_utils.to_categorical(y)
        self.model.fit(X, y, nb_epoch=self.nb_epoch, batch_size=self.batch_size,
                       validation_split=self.validation_split)

    def predict_proba(self, Xtest):
        ypred = self.model.predict_proba(Xtest, batch_size=self.batch_size, verbose=self.verbose)
        return ypred

    def predict(self, Xtest):
        ypred = self.model.predict(Xtest, batch_size=self.batch_size, verbose=self.verbose)
        return np_utils.probas_to_classes(ypred)


class KerasNNReg(BaseEstimator):
    def __init__(self, dims=66, nb_classes=1, nb_epoch=30, learning_rate=0.5, validation_split=0.0, batch_size=64,
                 loss='mean_absolute_error', verbose=1):
        self.dims = dims
        self.nb_classes = nb_classes
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.loss = loss
        self.verbose = verbose

        # Embedding
        # model_store = Sequential()
        # model_store.add(Embedding(1115, 50, input_length=1))
        # model_store.add(Reshape(dims=(50,)))
        # models.append(model_store)


        self.model = Sequential()
        # self.model.add(Merge(models, mode='concat'))
        # Keras model
        self.model.add(Dense(output_dim=1024, input_dim=dims, init='uniform'))
        self.model.add(Activation('sigmoid'))
        self.model.add(BatchNormalization())
        #self.model.add(Dropout(0.0))

        self.model.add(Dense(output_dim=512, init='uniform'))
        self.model.add(Activation('sigmoid'))
        self.model.add(BatchNormalization())
        #self.model.add(Dropout(0.0))

        #self.model.add(Dense(output_dim=256, init='uniform'))
        #self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.1))

        self.model.add(Dense(1))
        # self.model.add(Activation('sigmoid'))

        print('Compiling Keras Deep Net with loss: %s' % (str(loss)))
        self.model.compile(loss=loss, optimizer='rmsprop')

    def fit(self, X, y, sample_weight=None):
        print('Fitting  Keras Deep Net for regression with batch_size %d, epochs %d  and learning rate: %f' % (
        self.batch_size, self.nb_epoch, self.learning_rate))
        self.model.fit(X, y, nb_epoch=self.nb_epoch, batch_size=self.batch_size,
                       validation_split=self.validation_split)

    def predict_proba(self, Xtest):
        ypred = self.model.predict_proba(Xtest, batch_size=self.batch_size, verbose=self.verbose)
        return ypred

    def predict(self, Xtest):
        ypred = self.model.predict(Xtest, batch_size=self.batch_size, verbose=self.verbose)
        return ypred

class KerasMP(BaseEstimator):
    def __init__(self, dims=66, nb_classes=1, nb_epoch=30, learning_rate=0.5, validation_split=0.0, batch_size=64,
                 loss='mean_absolute_error', verbose=1):
        self.dims = dims
        self.nb_classes = nb_classes
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.loss = loss
        self.verbose = verbose

        # Embedding
        # model_store = Sequential()
        # model_store.add(Embedding(1115, 50, input_length=1))
        # model_store.add(Reshape(dims=(50,)))
        # models.append(model_store)

        self.model = Sequential()
        # self.model.add(Merge(models, mode='concat'))
        # Keras model
        self.model.add(Dense(output_dim=1024, input_dim=dims, init='glorot_uniform'))
        self.model.add(Activation('sigmoid'))
        self.model.add(BatchNormalization())

        self.model.add(Dense(output_dim=512, init='glorot_uniform'))
        self.model.add(Activation('sigmoid'))
        self.model.add(BatchNormalization())

        self.model.add(Dense(1))
        # self.model.add(Activation('sigmoid'))

        print('Compiling Keras Deep Net with loss: %s' % (str(loss)))
        self.model.compile(loss=loss, optimizer='rmsprop')

    def fit(self, X, y, sample_weight=None):
        print('Fitting  Keras Deep Net for regression with batch_size %d, epochs %d  and learning rate: %f' % (
        self.batch_size, self.nb_epoch, self.learning_rate))
        self.model.fit(X, y, nb_epoch=self.nb_epoch, batch_size=self.batch_size,
                       validation_split=self.validation_split)

    def predict_proba(self, Xtest):
        ypred = self.model.predict_proba(Xtest, batch_size=self.batch_size, verbose=self.verbose)
        return ypred

    def predict(self, Xtest):
        ypred = self.model.predict(Xtest, batch_size=self.batch_size, verbose=self.verbose)
        return ypred