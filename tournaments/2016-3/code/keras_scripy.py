import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
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
test = pd.read_csv(test_file)
#train["cat"] = train.apply(remove_addition, axis=1)
#test["cat"] = test.apply(remove_addition, axis=1)
#train = train.drop("c1", axis=1)

# ==== Create more features from the current features we have ====

tr_features = list(train.columns)
#features.remove("validation")
tr_features.remove("target")

for f in tr_features:
    for g in tr_features:
        if f != g:
            if not (str(g) + "_" + str(f)) in train.columns:
                train[str(f) + "_" + str(g)] = train[f] * train[g]

te_features = list(test.columns)
te_features.remove("t_id")

for f in te_features:
    for g in te_features:
        if f != g:
            if not (str(g) + "_" + str(f)) in test.columns:
                test[str(f) + "_" + str(g)] = test[f] * test[g]


# ==== Splitting dataset into training and validation ====

#training = train[train["validation"] == 0]
#validation = train[train["validation"] == 1]
#training = training.drop("validation", axis=1)
#validation = validation.drop("validation", axis=1)

# ==== Standard scaling of the inputs ====

#train = train.drop(["validation", "target"], axis=1)
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
#validation_target = validation["target"].values.T.astype(np.int32)
#validation = validation.drop("target", axis=1)

#X = np.array(training).astype(np.float32)
X = np.array(train).astype(np.float32)
X = scaler1.transform(X).astype(np.float32)
#Y = np.array(validation).astype(np.float32)
Y = np.array(test).astype(np.float32)
#Y = scaler.transform(Y).astype(np.float32)
Y = scaler2.transform(Y).astype(np.float32)
# ==== Parameters ====

num_features = X.shape[1]
epochs = 2

print num_features
print Y.shape[1]

hidden_layers = 4
hidden_units = 1024
dropout_p = 0.75

val_auc = np.zeros(epochs)

# ==== Defining the neural network shape ====

#l_in = InputLayer(shape=(None, num_features))
#l_hidden1 = DenseLayer(l_in, num_units=hidden_units)
#l_hidden2 = DropoutLayer(l_hidden1, p=dropout_p)
#l_current = l_hidden2
#for k in range(hidden_layers - 1):
#    l_current = highway_dense(l_current)
#    l_current = DropoutLayer(l_current, p=dropout_p)
#l_dropout = DropoutLayer(l_current, p=dropout_p)
#l_out = DenseLayer(l_dropout, num_units=2, nonlinearity=softmax)

# ==== Neural network definition ====

#net1 = NeuralNet(layers=l_out,
#                 update=adadelta, update_rho=0.95, update_learning_rate=1.0,
#                 objective_loss_function=categorical_crossentropy,
#                 train_split=TrainSplit(eval_size=0), verbose=0, max_epochs=1)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
"""
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(hidden_units, input_dim=num_features))
model.add(Activation('tanh'))
model.add(Dropout(dropout_p))
model.add(Dense(num_features, hidden_units))
model.add(Activation('tanh'))
model.add(Dropout(dropout_p))
model.add(Dense(hidden_units, hidden_units/2))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
                      optimizer=sgd)
"""

model = Sequential()
model.add(Dense(hidden_units, input_dim=num_features, init='uniform', activation='relu'))
model.add(Dropout(dropout_p))
model.add(Dense(hidden_units, activation='relu'))
model.add(Dropout(dropout_p))
model.add(Dense(2, activation='sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# "class_mode" defaults to "categorical". For correctly displaying accuracy
# in a binary classification problem, it should be set to "binary".
model.compile(loss='binary_crossentropy',
                      optimizer=sgd,
                                    class_mode='binary')


# ==== Print out input shape for diagnosis ====
print len(training_target)
print(X.shape)
print(training_target.shape)

#cat_training_target, cat_validation_target = [np_utils.to_categorical(x) for x in (training_target, validation_target)]
#cat_training_target = [np_utils.to_categorical(x) for x in training_target]
cat_training_target = np_utils.to_categorical(training_target, 2)

# ==== Train it for n iterations and validate on each iteration ====

dataset_blend_test = np.zeros((test.shape[0], len(range(epochs))))

for i in range(epochs):
    print i
    model.fit(X, cat_training_target, batch_size=10000, nb_epoch=3)
    #pred = model.predict_proba(Y)[:, 1]
    #val_auc[i] = roc_auc_score(validation_target, pred)
    #print(i + 1, "\t", round(val_auc[i] * 100, 3), "\t", round(max(val_auc) * 100, 3), "\t")
    y_submission = model.predict_proba(Y)[:,1]
    dataset_blend_test[:, i] = y_submission


#dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

y_sub = dataset_blend_test.mean(1)
df = pd.DataFrame({'probability':y_sub})
header = ['probability']
    
print "Saving Results."
df.to_csv('NNtest.csv', columns = header, index=False)



    
#df = pd.DataFrame({'probability':y_submission})
#header = ['probability']

#print "Saving Results."
#df.to_csv('test.csv', columns = header, index=False)





# ==== Make of the plot of the validation accuracy per iteration ====

#pyplot.plot(val_auc, linewidth=2)
#pyplot.axhline(y=max(val_auc), xmin=0, xmax=epochs, c="blue", linewidth=3, zorder=0)
#pyplot.grid()
#pyplot.title("Maximum AUC is " + str(round(max(val_auc) * 100, 3)) + '%')
#pyplot.xlabel("Epoch")
#pyplot.ylabel("Validation AUC")
#pyplot.show()
