import numpy as np
import pandas as pd
from lasagne.init import Orthogonal, Constant
from lasagne.layers import DenseLayer, MergeLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import softmax, rectify, sigmoid
from lasagne.objectives import categorical_crossentropy, binary_crossentropy
from lasagne.updates import nesterov_momentum, adadelta
from matplotlib import pyplot
from nolearn.lasagne import NeuralNet, TrainSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class MultiplicativeGatingLayer(MergeLayer):
    def __init__(self, gate, input1, input2, **kwargs):
        incomings = [gate, input1, input2]
        super(MultiplicativeGatingLayer, self).__init__(incomings, **kwargs)
        assert gate.output_shape == input1.output_shape == input2.output_shape

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        return inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]


def highway_dense(incoming, Wh=Orthogonal(), bh=Constant(0.0),
                  Wt=Orthogonal(), bt=Constant(-4.0),
                  nonlinearity=rectify, **kwargs):
    num_inputs = int(np.prod(incoming.output_shape[1:]))

    l_h = DenseLayer(incoming, num_units=num_inputs, W=Wh, b=bh, nonlinearity=nonlinearity)
    l_t = DenseLayer(incoming, num_units=num_inputs, W=Wt, b=bt, nonlinearity=sigmoid)

    return MultiplicativeGatingLayer(gate=l_t, input1=l_h, input2=incoming)


def remove_addition(X):
    return int(str(X[14]).split("_")[1])

# ==== Read in the csv and correct the c1 column ====
train_file = '../../../numerai_datasets/numerai_training_data.csv'
test_file = '../../../numerai_datasets/numerai_tournament_data.csv'
train = pd.read_csv(train_file)
train["cat"] = train.apply(remove_addition, axis=1)
train = train.drop("c1", axis=1)

# ==== Create more features from the current features we have ====

features = list(train.columns)
features.remove("validation")
features.remove("target")

for f in features:
    for g in features:
        if f != g:
            if not (str(g) + "_" + str(f)) in train.columns:
                train[str(f) + "_" + str(g)] = train[f] * train[g]

# ==== Splitting dataset into training and validation ====

training = train[train["validation"] == 0]
validation = train[train["validation"] == 1]
training = training.drop("validation", axis=1)
validation = validation.drop("validation", axis=1)

# ==== Standard scaling of the inputs ====

train = train.drop(["validation", "target"], axis=1)
train = np.array(train).astype(np.float32)
scaler = StandardScaler()
scaler.fit(train)

training_target = training["target"].values.T.astype(np.int32)
training = training.drop("target", axis=1)
validation_target = validation["target"].values.T.astype(np.int32)
validation = validation.drop("target", axis=1)

X = np.array(training).astype(np.float32)
X = scaler.transform(X).astype(np.float32)
Y = np.array(validation).astype(np.float32)
Y = scaler.transform(Y).astype(np.float32)

# ==== Parameters ====

num_features = X.shape[1]
epochs = 10

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
print training_target[0]
print(X.shape)
print(training_target.shape)

cat_training_target, cat_validation_target = [np_utils.to_categorical(x) for x in (training_target, validation_target)]

# ==== Train it for n iterations and validate on each iteration ====

for i in range(epochs):
    print i
    model.fit(X, cat_training_target, batch_size=10000, nb_epoch=10)
    pred = model.predict_proba(Y)[:, 1]
    val_auc[i] = roc_auc_score(validation_target, pred)
    print(i + 1, "\t", round(val_auc[i] * 100, 3), "\t", round(max(val_auc) * 100, 3), "\t")


# ==== Make of the plot of the validation accuracy per iteration ====

pyplot.plot(val_auc, linewidth=2)
pyplot.axhline(y=max(val_auc), xmin=0, xmax=epochs, c="blue", linewidth=3, zorder=0)
pyplot.grid()
pyplot.title("Maximum AUC is " + str(round(max(val_auc) * 100, 3)) + '%')
pyplot.xlabel("Epoch")
pyplot.ylabel("Validation AUC")
pyplot.show()
