import sys
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adadelta, Adam, rmsprop
from keras.utils import np_utils, generic_utils
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

def remove_addition(X):
    return int(str(X[14]).split("_")[1])

# ==== Read in the csv and correct the c1 column ====
train_file = '../../../numerai_datasets_new/numerai_training_data.csv'
test_file = '../../../numerai_datasets_new/numerai_tournament_data.csv'
train = pd.read_csv(train_file)
copy_train = pd.read_csv(train_file)

# ==== Create more features from the current features we have ====

tr_features = list(train.columns)
#features.remove("validation")
tr_features.remove("target")

for f in tr_features:
    for g in tr_features:
        if f != g:
            if not (str(g) + "_" + str(f)) in train.columns:
                train[str(f) + "_" + str(g)] = train[f] * train[g]


# ==== Splitting dataset into training and validation ====

msk = np.random.rand(len(train)) < 0.8
training         = train[msk]
copy_training    = train[msk]
validation       = train[~msk]
copy_validation  = train[~msk]
print len(training)
print len(validation)

# ==== Standard scaling of the inputs ====

#train = train.drop(["validation", "target"], axis=1)
training = training.drop(["target"], axis=1)
training = np.array(training).astype(np.float32)
validation = validation.drop(["target"], axis=1)
validation = np.array(validation).astype(np.float32)
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler1.fit(training)
scaler2.fit(validation)

training_target = copy_training["target"].values.T.astype(np.int32)
training = copy_training.drop("target", axis=1)
validation_target = copy_validation["target"].values.T.astype(np.int32)
validation = copy_validation.drop("target", axis=1)

X = np.array(training).astype(np.float32)
X = scaler1.transform(X).astype(np.float32)
Y = np.array(validation).astype(np.float32)
Y = scaler2.transform(Y).astype(np.float32)

cat_training_target, cat_validation_target = [np_utils.to_categorical(x) for x in (training_target, validation_target)]
# ==== Parameters ====

num_features = X.shape[1]
epochs = 2

hidden_layers = 4
hidden_units = 1024
dropout_p = 0.75

val_auc = np.zeros(epochs)

# ==== Train it for n iterations and validate on each iteration ====

space = {'choice': hp.choice('num_layers', [ {'layers':'two', }, {'layers':'three', 'units3': hp.uniform('units3', 64,1024), 'dropout3': hp.uniform('dropout3', .25,.75)}]),'units1': hp.uniform('units1', 64,1024),'units2': hp.uniform('units2', 64,1024),'dropout1': hp.uniform('dropout1', .25,.75),'dropout2': hp.uniform('dropout2',  .25,.75),'batch_size' : hp.choice('batch_size', [5000,10000]),'nb_epochs' :  hp.choice('nb_epochs',[2, 10]),'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),'activation': 'relu'}


def NN(params):
    #model = Sequential()
    #model.add(Dense(hidden_units, input_dim=num_features, init='uniform', activation='relu'))
    #model.add(Dropout(dropout_p))
    #model.add(Dense(hidden_units, activation='relu'))
    #model.add(Dropout(dropout_p))
    #model.add(Dense(2, activation='sigmoid'))

    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # "class_mode" defaults to "categorical". For correctly displaying accuracy
    # in a binary classification problem, it should be set to "binary".
    #model.compile(loss='binary_crossentropy',
    #                      optimizer=sgd,
    #                                    class_mode='binary')
    print ('Params testing: ', params) 
    
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim = num_features)) 
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=params['units2'], init = "glorot_uniform")) 
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers']== 'three':
        model.add(Dense(output_dim=params['choice']['units3'], init = "glorot_uniform")) 
        model.add(Activation(params['activation']))
        model.add(Dropout(params['choice']['dropout3']))    

        model.add(Dense(2))
        model.add(Activation('sigmoid'))
    else:
        model.add(Dense(2))
        model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'], class_mode='binary')
    
    model.fit(X, cat_training_target, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose=0)
    pred_auc = model.predict_proba(Y, batch_size = 5000, verbose = 0)
    acc = roc_auc_score(cat_validation_target, pred_auc)
    loss = log_loss(cat_validation_target, pred_auc)
    print('AUC:', acc)
    print('log loss', loss)
    sys.stdout.flush() 
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(NN, space, algo=tpe.suggest, max_evals=50, trials=trials)
print 'best: '
print best
