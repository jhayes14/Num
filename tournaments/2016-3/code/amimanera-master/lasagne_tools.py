
#import sys
#sys.path.append("/usr/local/cuda-6.0/bin")

import theano
import numpy as np

from lasagne import layers
from lasagne import nonlinearities
from lasagne.updates import nesterov_momentum,sgd,rmsprop,adadelta,adagrad

from lasagne.objectives import Objective
from lasagne.regularization import l2
from lasagne.objectives import categorical_crossentropy

import random

#from lasagne.regularization import l1
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
#from BatchNormalization import *

import cPickle as pickle

def plotNN(net1):
    train_loss = np.array([i["train_loss"] for i in net1.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
    plt.plot(train_loss, linewidth=3, label="train")
    plt.plot(valid_loss, linewidth=3, label="valid")
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(1e-3, 1e-2)
    plt.yscale("log")
    plt.show()


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        #print 'NEW VALUE:',new_value
        getattr(nn, self.name).set_value(new_value)

def float32(k):
    return np.cast['float32'](k)


Maxout = layers.pool.FeaturePoolLayer


class L2Regularization(Objective):
  
    def __init__(self, input_layer, loss_function=None, aggregation='mean',**args):
	Objective.__init__(self, input_layer, loss_function, aggregation)
	self.alpha=args['alpha']
    
    def get_loss(self, input=None, target=None, deterministic=False, **kwargs):
        loss = super(L2Regularization, self).get_loss(input=input,target=target, deterministic=deterministic, **kwargs)
        if not deterministic:
            return loss + self.alpha * l2(self.input_layer)
        else:
            return loss

class RMSE(Objective):
  
    def __init__(self, input_layer, loss_function=None, aggregation='mean',**args):
	Objective.__init__(self, input_layer, loss_function, aggregation)
	if 'alpha' in args:
	  self.alpha=args['alpha']
	else:
	  self.alpha=0.0
    
    def get_loss(self, input=None, target=None, deterministic=False, **kwargs):
        loss = super(RMSE, self).get_loss(input=input,target=target, deterministic=deterministic, **kwargs)
        loss = loss**0.5 + self.alpha * l2(self.input_layer)
        return loss 


class MSE(Objective):
  
    def __init__(self, input_layer, loss_function=None, aggregation='mean',**args):
	Objective.__init__(self, input_layer, loss_function, aggregation)
    
    def get_loss(self, input=None, target=None, deterministic=False, **kwargs):
        loss = super(MSE, self).get_loss(input=input,target=target, deterministic=deterministic, **kwargs)
        return loss


def shuffle_arrays(*arrays):
    p = np.random.permutation(len(arrays[0]))
    return [array[p] for array in arrays]


class ShuffleBatchIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, X, y=None):
        self.X, self.y = X, y
        return self

    def __iter__(self):
        self.X, self.y = shuffle_arrays(self.X, self.y)
        n_samples = self.X.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) / bs):	    
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb



class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()
	  
	  
nnet2 = NeuralNet(

    layers=[ 
      ('input', layers.InputLayer),      
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer),
	],
    # layer parameters:
    input_shape=(None,92),  # 96x96 input pixels per batch

    hidden1_num_units=500,  # number of units in hidden layer
    hidden1_nonlinearity=nonlinearities.rectify,
    dropout1_p=0.5,

    hidden2_num_units=500,
    hidden2_nonlinearity=nonlinearities.rectify,
    dropout2_p=0.5,

    hidden3_num_units=500,
    hidden3_nonlinearity=nonlinearities.rectify,
    dropout3_p=0.5,

    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=9,  # 30 target values

    objective=L2Regularization,
    objective_alpha=0.000005,

    eval_size=0.0,
    batch_iterator_train=BatchIterator(batch_size=1024),
    batch_iterator_test=BatchIterator(batch_size=1024),

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    on_epoch_finished=[
	AdjustVariable('update_learning_rate', start=0.03, stop=0.01),
	AdjustVariable('update_momentum', start=0.9, stop=0.999),
	#EarlyStopping(patience=200),
	],


    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=100,  # we want to train this many epochs
    verbose=1,
    )

nnet3 = NeuralNet(

    layers=[ 
      ('input', layers.InputLayer),      
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('output', layers.DenseLayer),
	],
    # layer parameters:
    input_shape=(None,93),  # 96x96 input pixels per batch

    hidden1_num_units=800,  # number of units in hidden layer
    hidden1_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
    dropout1_p=0.5,

    hidden2_num_units=800,
    hidden2_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
    dropout2_p=0.5,

    hidden3_num_units=800,

    #hidden5_nonlinearity=nonlinearities.rectify,
    #dropout5_p=0.5,

    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=9,  # 30 target values

    objective=L2Regularization,
    objective_alpha=1E-9,

    eval_size=0.0,

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.01)),
    update_momentum=theano.shared(float32(0.9)),

    on_epoch_finished=[
	AdjustVariable('update_learning_rate', start=0.01, stop=0.001),
	#AdjustVariable('update_momentum', start=0.9, stop=0.999),
	#EarlyStopping(patience=200),
	],


    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=50,  # we want to train this many epochs
    verbose=1,
    )


nnet4 = NeuralNet(#0.465?

    layers=[ 
      ('input', layers.InputLayer), 
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	#('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer),
	],
    # layer parameters:
    input_shape=(None,93), 

    hidden1_num_units=800, 
    hidden1_nonlinearity=nonlinearities.rectify,
    dropout1_p=0.5,

    hidden2_num_units=800,
    hidden2_nonlinearity=nonlinearities.rectify,
    dropout2_p=0.5,

    hidden3_num_units=600,
    hidden3_nonlinearity=nonlinearities.rectify,
    #dropout3_p=0.25,

    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=9,  # 30 target values

    batch_iterator_train=ShuffleBatchIterator(batch_size = 32),
    #batch_iterator_train=BatchIterator(batch_size = 32),
    #objective=L2Regularization,
    #objective_alpha=0.00005,

    eval_size=0.125,

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.005)),
    update_momentum=theano.shared(float32(0.9)),

    on_epoch_finished=[
	AdjustVariable('update_learning_rate', start=0.005, stop=0.00001),
	#AdjustVariable('update_momentum', start=0.9, stop=0.999),
	#EarlyStopping(patience=200),
	],


    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=50,  # we want to train this many epochs
    verbose=1,
    )


nnet5 = NeuralNet(

    layers=[ 
      ('input', layers.InputLayer),      
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('output', layers.DenseLayer),
	],
    # layer parameters:
    input_shape=(None,93),  

    hidden1_num_units=500,  # number of units in hidden layer
    hidden1_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
    dropout1_p=0.5,

    hidden2_num_units=500,
    hidden2_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
    dropout2_p=0.5,

    hidden3_num_units=500,
    #hidden3_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),

    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=9,  # 30 target values

    objective=L2Regularization,
    objective_alpha=1E-9,

    eval_size=0.2,

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.005)),
    update_momentum=theano.shared(float32(0.9)),

    on_epoch_finished=[
	AdjustVariable('update_learning_rate', start=0.005, stop=0.0001),
	#AdjustVariable('update_momentum', start=0.9, stop=0.999),
	#EarlyStopping(patience=200),
	],


    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=100,  # we want to train this many epochs
    verbose=1,
    )

nnet1 = NeuralNet(
    layers=[ 
	('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('maxout1', Maxout),
        ('dropout1', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('maxout2', Maxout),
        ('dropout2', layers.DropoutLayer),
        ('hidden3', layers.DenseLayer),
        ('maxout3', Maxout),
        #('dropout3', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],

    # layer parameters:
    input_shape=(None,93),

    hidden1_num_units=600,  
    hidden1_nonlinearity=nonlinearities.identity,
    maxout1_ds=2,
    dropout1_p=0.5,
   
    
    hidden2_num_units=600, 
    hidden2_nonlinearity=nonlinearities.identity,
    maxout2_ds=2,
    dropout2_p=0.5,
    
    
    hidden3_num_units=600, 
    hidden3_nonlinearity=nonlinearities.identity,
    maxout3_ds=3,
    #dropout3_p=0.5,
    
    batch_iterator_train=BatchIterator(batch_size = 64),
    
    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=9, 

    eval_size=0.15,

    #objective=categorical_crossentropy,
    #objective_alpha=1E-6,
    #update=nesterov_momentum,
    update = sgd,
    update_learning_rate=theano.shared(float32(0.005)),
    update_momentum=theano.shared(float32(0.9)),

    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=100,  # we want to train this many epochs
    verbose=1,
    
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.005, stop=0.0000001),
        #AdjustVariable('update_momentum', start=0.9, stop=0.999),
        #EarlyStopping(patience=20),
        ],  
    )
    
nnet6 = NeuralNet(
    layers=[ 
	('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('wta1', layers.FeatureWTALayer),
        ('dropout1', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('wta2', layers.FeatureWTALayer),
        ('dropout2', layers.DropoutLayer),
        ('hidden3', layers.DenseLayer),
        #('wta3', layers.FeatureWTALayer),
        ('output', layers.DenseLayer),
        ],

    # layer parameters:
    input_shape=(None,93),

    hidden1_num_units=600,  
    hidden1_nonlinearity=nonlinearities.identity,
    wta1_ds=2,
    dropout1_p=0.5,
   
    
    hidden2_num_units=600, 
    hidden2_nonlinearity=nonlinearities.identity,
    wta2_ds=2,
    dropout2_p=0.5,
    
    
    hidden3_num_units=600, 
    #hidden3_nonlinearity=nonlinearities.identity,
    #wta3_ds=2,
    #dropout3_p=0.5,
    batch_iterator_train=ShuffleBatchIterator(batch_size = 32),
    #batch_iterator_train=BatchIterator(batch_size = 32),
    
    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
    output_num_units=9, 

    eval_size=0.15,

    #objective=categorical_crossentropy,
    #objective_alpha=1E-6,
    update=nesterov_momentum,
    #update = sgd,
    update_learning_rate=theano.shared(float32(0.005)),
    update_momentum=theano.shared(float32(0.9)),

    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=100,  # we want to train this many epochs
    verbose=1,
    
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.005, stop=0.00000001),
        #AdjustVariable('update_momentum', start=0.9, stop=0.999),
        #EarlyStopping(patience=20),
        ],  
    )    

#http://didericksen.github.io/tswift/
nnet7 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('dropout1', layers.DropoutLayer),
        
        ('hidden2', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        
        ('hidden3', layers.DenseLayer),
        ('dropout3', layers.DropoutLayer),
        
        ('hidden4', layers.DenseLayer),
        #('dropout4', layers.DropoutLayer),
        
        ('output', layers.DenseLayer)
        ],
    input_shape = (None, 98),
    dropout1_p = 0.1,
    
    hidden2_num_units = 600,
    dropout2_p = 0.5,
    
    hidden3_num_units = 600,
    dropout3_p = 0.5,
    
    hidden4_num_units = 500,
    #dropout4_p = 0.5,
    
    output_num_units = 9,
    output_nonlinearity = nonlinearities.softmax,

    eval_size=0.15,

    batch_iterator_train=ShuffleBatchIterator(batch_size = 64),
    # optimization method:
    update = adagrad,
    update_learning_rate=theano.shared(float32(0.03)),
    
    regression = False, 
    max_epochs = 50,
    verbose = 1,
    
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        #EarlyStopping(patience=20),
        ],   
    )



#https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/13851/lasagne-with-2-hidden-layers
nnet8 = NeuralNet(layers=[('input', layers.InputLayer),#0.454
('dropout0', layers.DropoutLayer),
('hidden1', layers.DenseLayer),
('dropout1', layers.DropoutLayer),
('hidden2', layers.DenseLayer),
('dropout2', layers.DropoutLayer), 
('output', layers.DenseLayer)],

input_shape=(None, 98),
dropout0_p=0.15,
hidden1_num_units=800,

dropout1_p=0.25,
hidden2_num_units=600,

dropout2_p=0.20,

output_num_units=9,
output_nonlinearity=nonlinearities.softmax,

#batch_iterator_train=ShuffleBatchIterator(batch_size = 32),

#update=nesterov_momentum,
update=adagrad,
update_learning_rate=theano.shared(float32(0.01)),
#update_momentum=0.9, only used with nesterov_
eval_size=0.0,
verbose=1,
max_epochs=150,

 on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.001),
        #EarlyStopping(patience=20),
        ],
)
 
#https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14134/deep-learning-h2o-0-44
nnet9 = NeuralNet(layers=[('input', layers.InputLayer),#0.464
('dropout0', layers.DropoutLayer),
('hidden1', layers.DenseLayer),
('dropout1', layers.DropoutLayer),
('hidden2', layers.DenseLayer),
('dropout2', layers.DropoutLayer), 
('hidden3', layers.DenseLayer),
('dropout3', layers.DropoutLayer), 
('output', layers.DenseLayer)],

input_shape=(None, 93),
dropout0_p=0.05,

hidden1_num_units=900,
hidden1_nonlinearity=nonlinearities.rectify,
dropout1_p=0.5,

hidden2_num_units=500,
hidden2_nonlinearity=nonlinearities.rectify,
dropout2_p=0.25,

hidden3_num_units=250,
hidden3_nonlinearity=nonlinearities.rectify,
dropout3_p=0.25,

output_num_units=9,
output_nonlinearity=nonlinearities.softmax,

objective=L2Regularization,
objective_alpha=1E-6,

#batch_iterator_train=ShuffleBatchIterator(batch_size = 32),

#update=nesterov_momentum,
update=adagrad,
update_learning_rate=theano.shared(float32(0.02)),
#update_momentum=0.9, only used with nesterov_
eval_size=0.0,
verbose=1,
max_epochs=100,

 on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.02, stop=0.001),
        #EarlyStopping(patience=20),
        ],
)
 

nnet_cater = NeuralNet(layers=[('input', layers.InputLayer),
('dropout0', layers.DropoutLayer),
('hidden1', layers.DenseLayer),
('dropout1', layers.DropoutLayer),
('hidden2', layers.DenseLayer),
('dropout2', layers.DropoutLayer),
#('hidden3', layers.DenseLayer),
#('dropout3', layers.DropoutLayer), 
('output', layers.DenseLayer)],

input_shape=(None, 260),
dropout0_p=0.0,

hidden1_num_units=256,
#hidden1_nonlinearity=nonlinearities.rectify,#not stable why?
hidden1_nonlinearity=nonlinearities.sigmoid,
#hidden1_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
#hidden1_nonlinearity=nonlinearities.tanh,
dropout1_p=0.15,

hidden2_num_units=256,
#hidden2_nonlinearity=nonlinearities.rectify,
hidden2_nonlinearity=nonlinearities.sigmoid,
#hidden2_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
#hidden2_nonlinearity=nonlinearities.tanh,
dropout2_p=0.15,

#hidden3_num_units=512,
#hidden2_nonlinearity=nonlinearities.rectify,
#hidden3_nonlinearity=nonlinearities.sigmoid,

output_num_units=1,
output_nonlinearity=None,

regression=True,
objective=RMSE,
objective_alpha=0.0,
batch_iterator_train=ShuffleBatchIterator(batch_size = 64),

#update=nesterov_momentum,
update=rmsprop,#0.005
#update=adagrad,#0.25
#update_learning_rate=theano.shared(float32(0.25)),
update_learning_rate=theano.shared(float32(0.01)),
#update_momentum=0.9, #only used with nesterov_

eval_size=0.2,
verbose=1,
max_epochs=75,

 on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
        #EarlyStopping(patience=20),
        ],
)

nnet_cater2 = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 271),
	dropout0_p=0.0,

	hidden1_num_units=512,
	hidden1_nonlinearity=nonlinearities.rectify,
	dropout1_p=0.0,

	hidden2_num_units=512,
	hidden2_nonlinearity=nonlinearities.rectify,
	dropout2_p=0.2,
	
	hidden3_num_units=512,
	hidden3_nonlinearity=nonlinearities.rectify,
	dropout3_p=0.2,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(0.001)),
	
	eval_size=0.0,
	verbose=1,
	max_epochs=50,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.001, stop=0.00005),
		#EarlyStopping(patience=20),
		],
)

nnet_cater3 = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 274),
	dropout0_p=0.0,

	hidden1_num_units=600,
	hidden1_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout1_p=0.10,

	hidden2_num_units=600,
	hidden2_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout2_p=0.10,
	
	hidden3_num_units=600,
	hidden3_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout3_p=0.10,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	objective_alpha=1.0*1E-3,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),#->32?

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(1.2e-03)),
	
	eval_size=0.0,
	verbose=1,
	max_epochs=75,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=1.2e-03, stop=0.00005),
		#EarlyStopping(patience=20),
		],
)



nnet_ensembler_rossmann = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 12),
	dropout0_p=0.0,

	hidden1_num_units=32,
	hidden1_nonlinearity=nonlinearities.rectify,
	dropout1_p=0.0,

	hidden2_num_units=32,
	hidden2_nonlinearity=nonlinearities.rectify,
	dropout2_p=0.0,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	objective_alpha=0.001,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(0.001)),
	
	eval_size=0.0,
	verbose=1,
	max_epochs=20,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.001, stop=0.00005),
		#EarlyStopping(patience=20),
		],
)

nnet_ensembler2 = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 26),
	dropout0_p=0.0,

	hidden1_num_units=128,
	hidden1_nonlinearity=nonlinearities.rectify,
	dropout1_p=0.0,

	hidden2_num_units=128,
	hidden2_nonlinearity=nonlinearities.rectify,
	dropout2_p=0.0,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	objective_alpha=0.001,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 32),

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(0.0025)),

	eval_size=0.0,
	verbose=1,
	max_epochs=75,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.0025, stop=0.00005),
		#EarlyStopping(patience=20),
		],
)

	
	
nnet_ensembler3 = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 29),
	dropout0_p=0.0,

	hidden1_num_units=128,
	hidden1_nonlinearity=nonlinearities.rectify,
	dropout1_p=0.0,

	hidden2_num_units=128,
	hidden2_nonlinearity=nonlinearities.rectify,
	dropout2_p=0.0,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	objective_alpha=0.001,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(0.002)),

	eval_size=0.0,
	verbose=1,
	max_epochs=75,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.002, stop=0.00005),
		#EarlyStopping(patience=20),
		],
)

nnet_ensembler4 = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 29),
	dropout0_p=0.0,

	hidden1_num_units=128,
	#hidden1_nonlinearity=nonlinearities.rectify,
    hidden1_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout1_p=0.0,

	hidden2_num_units=128,
	#hidden2_nonlinearity=nonlinearities.rectify,
    hidden2_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout2_p=0.0,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	objective_alpha=0.001,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(0.002)),

	eval_size=0.0,
	verbose=1,
	max_epochs=75,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.002, stop=0.00005),
		#EarlyStopping(patience=20),
		],
)

nnet_ensembler5 = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 29),
	dropout0_p=0.0,

	hidden1_num_units=128,
	hidden1_nonlinearity=nonlinearities.rectify,
	dropout1_p=0.0,

	hidden2_num_units=128,
	hidden2_nonlinearity=nonlinearities.rectify,
	dropout2_p=0.0,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	objective_alpha=0.001,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 32),

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(0.002)),

	eval_size=0.0,
	verbose=1,
	max_epochs=75,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.002, stop=0.00005),
		#EarlyStopping(patience=20),
		],
)

nnet_ensembler6 = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 33),
	dropout0_p=0.0,

	hidden1_num_units=128,
	hidden1_nonlinearity=nonlinearities.rectify,
	dropout1_p=0.0,

	hidden2_num_units=128,
	hidden2_nonlinearity=nonlinearities.rectify,
	dropout2_p=0.0,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	objective_alpha=0.001,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(0.002)),

	eval_size=0.0,
	verbose=1,
	max_epochs=75,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.002, stop=0.00005),
		#EarlyStopping(patience=20),
		],
)


nn10_BN_small2 = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	#('hidden3', layers.DenseLayer),
	#('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 258),
	dropout0_p=0.0,

	hidden1_num_units=500,
	#hidden1_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
    hidden1_nonlinearity=nonlinearities.rectify,
	dropout1_p=0.1,

	hidden2_num_units=500,
	hidden2_nonlinearity=nonlinearities.rectify,
    #hidden1_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout2_p=0.1,

	#hidden3_num_units=600,
	#hidden3_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	#dropout3_p=0.0,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	objective_alpha=1.0*1E-5,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 256),#->32?

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(1.0*1e-04)),

	eval_size=0.2,
	verbose=1,
	max_epochs=75,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=1.0*1e-04, stop=1.0*1e-06),
		#EarlyStopping(patience=20),
		],
)

nn10_BN_small = NeuralNet(layers=[('input', layers.InputLayer),
    ('dropout0', layers.DropoutLayer),
    ('hidden1', layers.DenseLayer),
    ('dropout1', layers.DropoutLayer),
    ('hidden2', layers.DenseLayer),
    ('dropout2', layers.DropoutLayer),
    #('hidden3', layers.DenseLayer),
    #('dropout3', layers.DropoutLayer), 
    ('output', layers.DenseLayer)],

    input_shape=(None, 260),
    dropout0_p=0.0,

    hidden1_num_units=500,
    #hidden1_nonlinearity=nonlinearities.rectify,#not stable why?
    hidden1_nonlinearity=nonlinearities.sigmoid,
    #hidden1_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
    #hidden1_nonlinearity=nonlinearities.tanh,
    dropout1_p=0.1,

    hidden2_num_units=500,
    #hidden2_nonlinearity=nonlinearities.rectify,
    hidden2_nonlinearity=nonlinearities.sigmoid,
    #hidden2_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
    #hidden2_nonlinearity=nonlinearities.tanh,
    dropout2_p=0.1,

    #hidden3_num_units=512,
    #hidden2_nonlinearity=nonlinearities.rectify,
    #hidden3_nonlinearity=nonlinearities.sigmoid,

    output_num_units=1,
    output_nonlinearity=None,

    regression=True,
    objective=RMSE,
    objective_alpha=1E-5,
    batch_iterator_train=BatchIterator(batch_size = 32),

    #update=nesterov_momentum,
    update=rmsprop,#0.005
    #update=adagrad,#0.25
    #update_learning_rate=theano.shared(float32(0.25)),
    update_learning_rate=theano.shared(float32(0.005)),
    #update_momentum=0.9, #only used with nesterov_

    eval_size=0.2,
    verbose=1,
    max_epochs=75,

    on_epoch_finished=[
	    AdjustVariable('update_learning_rate', start=0.005, stop=0.00001),
	    #EarlyStopping(patience=20),
	    ],
)

ross1 = NeuralNet(layers=[('input', layers.InputLayer),
	('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, 24),
	dropout0_p=0.0,

	hidden1_num_units=256,
	hidden1_nonlinearity=nonlinearities.rectify,
	dropout1_p=0.1,

	hidden2_num_units=256,
	hidden2_nonlinearity=nonlinearities.rectify,
	dropout2_p=0.1,

	hidden3_num_units=256,
	hidden3_nonlinearity=nonlinearities.rectify,
	dropout3_p=0.1,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	#objective_alpha=1.0*1E-3,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),#->32?

	update=rmsprop,
	update_learning_rate=theano.shared(float32(1E-05)),

	eval_size=0.5,
	verbose=1,
	max_epochs=75,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=1e-05, stop=0.0005),
		#EarlyStopping(patience=20),
		],
)