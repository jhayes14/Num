from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adadelta, Adam, rmsprop
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils, generic_utils
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import f_blend
import warnings

warnings.filterwarnings("ignore")

# ==== Read in the csv and correct the c1 column ====
train_file = '../../../numerai_datasets_new/numerai_training_data.csv'
test_file = '../../../numerai_datasets_new/numerai_tournament_data.csv'
train = pd.read_csv(train_file)
copy_train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

# ==== Create more features from the current features we have ====

tr_features = list(train.columns)
tr_features.remove("target")

for f in tr_features:
    for g in tr_features:
        if f != g:
            if not (str(g) + "_" + str(f)) in train.columns:
                train[str(f) + "_" + str(g)] = train[f] * train[g]
#        elif f == g:
#            if not (str(g) + "_" + str(f)) in train.columns:
#                train[str(f) + "_" + str(g)] = train[f] * train[g]

train['f_mean'] = train.mean(axis=1)
train['f_std'] = train.std(axis=1)
test['f_mean'] = test.mean(axis=1)
test['f_std'] = test.std(axis=1)


te_features = list(test.columns)
te_features.remove("t_id")

for f in te_features:
    for g in te_features:
        if f != g:
            if not (str(g) + "_" + str(f)) in test.columns:
                test[str(f) + "_" + str(g)] = test[f] * test[g]
#        elif f == g:
#            if not (str(g) + "_" + str(f)) in test.columns:
#                test[str(f) + "_" + str(g)] = test[f] * test[g]

blend_train, blend_test = f_blend.blending()

for i in range(len(blend_train[0])):
    train[str(i)] = blend_train[:,i]
    test[str(i)]  = blend_test[:,i]


# ==== Standard scaling of the inputs ====

train = train.drop(["target"], axis=1)
train = np.array(train).astype(np.float32)
test = test.drop(["t_id"], axis=1)
test = np.array(test).astype(np.float32)
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler1.fit(train)
scaler2.fit(test)

training_target = copy_train["target"].values.T.astype(np.int32)
training = copy_train.drop("target", axis=1)

X = np.array(train).astype(np.float32)
X = scaler1.transform(X).astype(np.float32)
Y = np.array(test).astype(np.float32)
Y = scaler2.transform(Y).astype(np.float32)

# ==== Parameters ====

num_features = X.shape[1]
cycles = 10
params = {'units1': 1024, 'units3': 512, 'units2': 2048, 'optimizer': 'adadelta', 'dropout3': 0.5, 'batch_size': 1000, 'num_layers': 3, 'nb_epochs': 10, 'dropout2': 0.2, 'dropout1': 0.8, 'activation': 'relu'}
#hidden_layers = 4
#hidden_units = 1024
#dropout_p = 0.75
val_auc = np.zeros(cycles)
cat_training_target = np_utils.to_categorical(training_target, 2)

# ==== Defining the neural network shape ====

##M1##
#model = Sequential()
#model.add(Dense(hidden_units, input_dim=num_features, init='uniform', activation='relu'))
#model.add(Dropout(dropout_p))
#model.add(Dense(hidden_units, activation='relu'))
#model.add(Dropout(dropout_p))
#model.add(Dense(2, activation='sigmoid'))
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# "class_mode" defaults to "categorical". For correctly displaying accuracy
# in a binary classification problem, it should be set to "binary".
#model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')

##M2##
model = Sequential()
model.add(Dense(output_dim=params['units1'], input_dim = num_features)) 
model.add(Activation(params['activation']))
#model.add(BatchNormalization())
model.add(Dropout(params['dropout1']))
model.add(Dense(output_dim=params['units2'], init = "glorot_uniform")) 
model.add(Activation(params['activation']))
#model.add(BatchNormalization())
model.add(Dropout(params['dropout2']))
model.add(Dense(output_dim=params['units3'], init = "glorot_uniform")) 
model.add(Activation(params['activation']))
#model.add(BatchNormalization())
model.add(Dropout(params['dropout2']))    
model.add(Dense(2))
model.add(Activation('sigmoid'))

##M3##
#model = Sequential()
#model.add(Dense(512, input_dim=num_features))
#model.add(PReLU())
#model.add(BatchNormalization())
#model.add(Dropout(0.5))

#model.add(Dense(512))
#model.add(PReLU())
#model.add(BatchNormalization())
#model.add(Dropout(0.5))

#model.add(Dense(512))
#model.add(PReLU())
#model.add(BatchNormalization())
#model.add(Dropout(0.5))

#model.add(Dense(2))
#model.add(Activation('softmax'))

sgd = SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')
#model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')
#model.compile(loss='binary_crossentropy', optimizer=params['optimizer'], class_mode='binary')

# ==== Train it for n iterations and validate on each iteration ====

dataset_blend_test = np.zeros((test.shape[0], len(range(cycles))))

for i in range(cycles):
    print i
    model.fit(X, cat_training_target, batch_size=params['batch_size'], nb_epoch=params['nb_epochs'], verbose=1)
    y_submission = model.predict_proba(Y, batch_size=params['batch_size'], verbose=1)[:,1]
    dataset_blend_test[:, i] = y_submission

#y_sub = dataset_blend_test[:, -1]
y_sub = dataset_blend_test.mean(1)
df = pd.DataFrame({'probability':y_sub})
header = ['probability']
    
print "Saving Results."
preds = 'NNtest-' + str(cycles) + '-cycles.csv'
df.to_csv(preds, columns = header, index=False)
test = pd.read_csv( test_file )
preds = pd.read_csv( preds )
test_ids = test[['t_id']]
#print preds.iloc[:10]
#print test_ids.iloc[0]
total = pd.concat(( test_ids, preds), axis = 1)
total.to_csv('NNout-' + str(cycles) + '-cycles.csv', index=False)


# ==== Make of the plot of the validation accuracy per iteration ====

#pyplot.plot(val_auc, linewidth=2)
#pyplot.axhline(y=max(val_auc), xmin=0, xmax=epochs, c="blue", linewidth=3, zorder=0)
#pyplot.grid()
#pyplot.title("Maximum AUC is " + str(round(max(val_auc) * 100, 3)) + '%')
#pyplot.xlabel("Epoch")
#pyplot.ylabel("Validation AUC")
#pyplot.show()
