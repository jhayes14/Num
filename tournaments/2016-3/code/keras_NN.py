from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adadelta, Adam, rmsprop
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils, generic_utils
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
#import f_select
import warnings
import dill
import os

warnings.filterwarnings("ignore")

# ==== Load Data ====

test_file = '../../../numerai_datasets_new/numerai_tournament_data.csv'
training_file = '../../../numerai_datasets_new/numerai_training_data.csv'
for n in [1,2]:
    print n
    feature_file = 'features__' + str(n) + '.pickle'
    if os.path.exists(feature_file):
        with open(feature_file, "rb") as input_file:
            D = dill.load(input_file)

            X               = D[0]
            training_target = D[1]
            Y               = D[2]
            test_id         = D[3]
            Xall            = D[4]

            # ==== Parameters ====

            num_features = X.shape[1]
            cycles = 3

            ## Model A params ##

            paramsA = {'units1': [1024, 512, 100], 'units2': [1024, 512, 50], \
                       'units3': [2048, 1024, 200], 'optimizer': ['adam','adam','adam'], \
                       'dropout1' : [0.8, 0.8, 0.3], 'dropout2' : [0.2, 0.2, 0.2], \
                       'dropout3': [0.5, 0.5, 0.1], 'batch_size': [1000, 1000, 1000], \
                       'nb_epochs': [5, 5, 5], 'activation': ['relu','relu','relu']} #'relu'

            #--------------#

            val_auc = np.zeros(cycles)
            cat_training_target = np_utils.to_categorical(training_target, 2)

            # ==== Defining the neural network model/shape ====

            def build_modelA(params, i):
                modelA = Sequential()
                modelA.add(Dense(output_dim=params['units1'][i], input_dim = num_features))
                modelA.add(Activation(params['activation'][i]))
                modelA.add(LeakyReLU())
                #modelA.add(BatchNormalization())
                modelA.add(Dropout(params['dropout1'][i]))
                modelA.add(Dense(output_dim=params['units2'][i], init = "glorot_uniform"))
                modelA.add(Activation(params['activation'][i]))
                modelA.add(PReLU())
                #modelA.add(BatchNormalization())
                modelA.add(Dropout(params['dropout2'][i]))
                modelA.add(Dense(output_dim=params['units3'][i], init = "glorot_uniform"))
                modelA.add(Activation(params['activation'][i]))
                modelA.add(PReLU())
                #modelA.add(BatchNormalization())
                modelA.add(Dropout(params['dropout3'][i]))
                modelA.add(Dense(2))
                modelA.add(Activation('sigmoid'))

                sgd = SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)
                modelA.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')
                #modelA.compile(loss='binary_crossentropy', optimizer=params['optimizer'], class_mode='binary')
                return modelA

            # ==== Train it for n cycles and validate on each iteration ====

            dataset_blend_test = np.zeros((Y.shape[0], len(range(cycles))))

            for i in range(cycles):
                print i
                modelA = build_modelA(paramsA, i)
                modelA.fit(X, cat_training_target, batch_size=paramsA['batch_size'][i], nb_epoch=paramsA['nb_epochs'][i], verbose=1)
                y_submission = modelA.predict_proba(Y, batch_size=paramsA['batch_size'][i], verbose=1)[:,1]
                print y_submission
                dataset_blend_test[:, i] = y_submission



            # ==== Create submission ====

            #y_sub = dataset_blend_test[:, -1]
            y_sub = dataset_blend_test.mean(1)
            df = pd.DataFrame({'probability':y_sub})
            header = ['probability']

            print "Saving Results."
            preds = 'New_NNtest-' + str(cycles) + '-cycles-' + str(n) + '_feature.csv'
            df.to_csv(preds, columns = header, index=False)
            test = pd.read_csv( test_file )
            preds = pd.read_csv( preds )
            test_ids = test[['t_id']]
            #print preds.iloc[:10]
            #print test_ids.iloc[0]
            total = pd.concat(( test_ids, preds), axis = 1)
            total.to_csv('New_NNout-' + str(cycles) + '-cycles-' + str(n) + '_feature.csv', index=False)


            # ==== Make of the plot of the validation accuracy per iteration ====

            #pyplot.plot(val_auc, linewidth=2)
            #pyplot.axhline(y=max(val_auc), xmin=0, xmax=epochs, c="blue", linewidth=3, zorder=0)
            #pyplot.grid()
            #pyplot.title("Maximum AUC is " + str(round(max(val_auc) * 100, 3)) + '%')
            #pyplot.xlabel("Epoch")
            #pyplot.ylabel("Validation AUC")
            #pyplot.show()
