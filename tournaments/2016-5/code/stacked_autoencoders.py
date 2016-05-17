from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import containers, noise
from keras.layers.core import Dense, AutoEncoder, Dropout
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.layers.noise import GaussianNoise
from keras.callbacks import EarlyStopping, Callback
#import hacking_script
from sklearn import preprocessing, metrics
from sklearn.cross_validation import train_test_split
import dill

"""def draw_stacked_encoders(X, batch_size=100, nb_epoch=3, hidden_layers=[42 , 100, 30, 20, 80, 10, 6, 3],
    noise_schedule=[0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0], nb_classes=1):
    X_train_tmp = X
    trained_encoders = []
    trained_decoders = []
    for n_in, n_out, sigma_noise in zip(hidden_layers[:-1], hidden_layers[1:], noise_schedule):
        print('Pre-training the layer: Input 42 -> Output {}'.format(n_out))
        ae = Sequential()
        encoder = containers.Sequential([noise.GaussianNoise(sigma_noise, input_shape=(n_in,)), Dense(output_dim=n_out, input_dim=n_in, activation='relu')])
        decoder = containers.Sequential([Dense(output_dim=n_in, input_dim=n_out, activation='relu')])
        ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
        ae.compile(loss='mean_squared_error', optimizer='adagrad')
        ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch)
        # Store trainined weight
        trained_encoders.append((ae.layers[0].encoder.layers[1], ae.layers[0].encoder.layers[1].get_weights()))
        trained_decoders.append((ae.layers[0].decoder, ae.layers[0].decoder.get_weights()))
        # Update training data
        X_train_tmp = ae.predict(X_train_tmp)
        print(X_train_tmp.shape)
    return trained_encoders


def create_model():
    #X_train, y_train, X_test, y_test = hacking_script.load_golden_data()
    feature_file = '/Users/jamie/Numerai/tournaments/2016-3/code/features__2.pickle'
    with open(feature_file, "rb") as input_file:
            D = dill.load(input_file)
            X               = D[0]
            training_target = D[1]
            Y               = D[2]
            test_id         = D[3]
    X_train, X_test, y_train, y_test = train_test_split(X, training_target, test_size=0.33, random_state=4)

    #X_live = hacking_script.load_unsupervised_data("numerai_live_data.csv", golden=True)
    #X_tournament = hacking_script.load_unsupervised_data("/Users/jamie/Numerai/numerai_datasets_new/numerai_tournament_data.csv", golden=True)
    X_all = np.vstack((X_train, Y, X_test))
    print(X_all.shape)
    trained_encoders = draw_stacked_encoders(X_all, nb_epoch=1, batch_size=100)

    print('Fine-tuning')
    model = Sequential([noise.GaussianNoise(0.5, input_shape=(42,))])
    print(len(trained_encoders))
    #print(zip(trained_encoders, [0.5, 0.4, 0.3, 0.2, 0.2, 0.15,0.1, 0, 0]))
    for ((encoder, weights), sigma_noise) in zip(trained_encoders, [0.5, 0.4, 0.3, 0.2, 0.2, 0.15,0.1, 0, 0]):
        model.add(encoder)
        #model.layers[-1].set_weights(weights)
        model.add(noise.GaussianNoise(sigma_noise))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adagrad')
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)

    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)

    history = model.fit(X_train, y_train, batch_size=100, nb_epoch=10,
                  show_accuracy=True, validation_data=(X_test, y_test), callbacks=[early_stopping])
    return history, model

if __name__ == "__main__":
    #X = np.random.rand(5000,42)
    #draw_stacked_encoders(X)
    #history, model = create_model()
    #X_train, y_train, X_test, y_test = hacking_script.load_golden_data()
    #feature_file = '/Users/jamie/Numerai/tournaments/2016-3/code/features__2.pickle'
    #with open(feature_file, "rb") as input_file:
    #        D = dill.load(input_file)
    #        X               = D[0]
    #        training_target = D[1]
    #X_train, X_test, y_train, y_test = train_test_split(X, training_target, test_size=0.33, random_state=4)
    #X_all = np.vstack((X_train, X_test))
    #print(X_all.shape)
    #trained_encoders = draw_stacked_encoders(X_all, nb_epoch=1, batch_size=100)

    #predictions = model.predict_proba(X_test)
    #y_train = np_utils.to_categorical(y_train, 2)
    #y_test = np_utils.to_categorical(y_test, 2)
    #print(metrics.roc_auc_score(y_test, predictions))
    #print(history.history)
    #hacking_script.create_tournament_entry(model, 'entry_nn')"""


"""from keras.layers import containers, AutoEncoder, Dense
from keras import models

X_train = np.random.rand(5000,32)
X_test = np.random.rand(200,32)


# input shape: (nb_samples, 32)
encoder = containers.Sequential([Dense(16, input_dim=32), Dense(8)])
decoder = containers.Sequential([Dense(16, input_dim=8), Dense(32)])

autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True)
model = models.Sequential()
model.add(autoencoder)

# training the autoencoder:
model.compile(optimizer='sgd', loss='mse')
model.fit(X_train, X_train, nb_epoch=10)

# predicting compressed representations of inputs:
autoencoder.output_reconstruction = False  # the model has to be recompiled after modifying this property
model.compile(optimizer='sgd', loss='mse')
representations = model.predict(X_test)

# the model is still trainable, although it now expects compressed representations as targets:
model.fit(X_test, representations, nb_epoch=1)  # in this case the loss will be 0, so it's useless

# to keep training against the original inputs, just switch back output_reconstruction to True:
autoencoder.output_reconstruction = True
model.compile(optimizer='sgd', loss='mse')
model.fit(X_train, X_train, nb_epoch=10)"""



def draw_stacked_encoders(X, batch_size=100, nb_epoch=3, hidden_layers=[42 , 100, 30, 20, 80, 10, 6, 3],
    noise_schedule=[0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0], nb_classes=1):
    X_train_tmp = X
    trained_encoders = []
    trained_decoders = []
    for n_in, n_out, sigma_noise in zip(hidden_layers[:-1], hidden_layers[1:], noise_schedule):
        print('Pre-training the layer: Input {} -> Output {}'.format(n_in,n_out))
        ae = Sequential()
        encoder = containers.Sequential([noise.GaussianNoise(sigma_noise, input_shape=(n_in,)), Dense(output_dim=n_out, input_dim=n_in, activation='tanh')])
        decoder = containers.Sequential([Dense(output_dim=n_in, input_dim=n_out, activation='tanh')])
        ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))
        ae.compile(loss='mean_squared_error', optimizer='adagrad')
        ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1)
        # Store trainined weight
        trained_encoders.append((ae.layers[0].encoder.layers[1], ae.layers[0].encoder.layers[1].get_weights()))
        trained_decoders.append((ae.layers[0].decoder, ae.layers[0].decoder.get_weights()))
        # Update training data
        ae.layers[0].output_reconstruction = False
        ae.compile(loss='mean_squared_error', optimizer='adagrad')
        X_train_tmp = ae.predict(X_train_tmp)
        print(X_train_tmp.shape)
    return trained_encoders

if __name__ == "__main__":
    X = np.random.rand(5000,42)
    draw_stacked_encoders(X)
