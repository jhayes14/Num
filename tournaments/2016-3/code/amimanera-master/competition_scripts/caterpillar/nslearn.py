import os
import errno
import numpy as np
from sklearn.preprocessing import LabelEncoder
import nolearn
import nolearn.lasagne
from lasagne.objectives import Objective
from lasagne.regularization import l2

from time import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def nint(x):
    return int(round(x,0))


class RMSE(Objective):
    def __init__(self, input_layer, loss_function=None, aggregation='mean',**args):
        Objective.__init__(self, input_layer, loss_function, aggregation)
        if args['alpha']:
            self.alpha=args['alpha']
        else:
            self.alpha=0.0

    def get_loss(self, input=None, target=None, deterministic=False, **kwargs):
        loss = super(RMSE, self).get_loss(input=input,target=target, deterministic=deterministic, **kwargs)
        loss = loss**0.5 + self.alpha * l2(self.input_layer)
        return loss


class BatchIterator(nolearn.lasagne.BatchIterator):
    def __init__(self, batch_size, subsample=1.0, seed=None):
        nolearn.lasagne.BatchIterator.__init__(self, batch_size)
        self.subsample = subsample
        self.random = np.random.RandomState(seed=seed)

    def transform(self, Xb, yb):
        size = len(Xb)
        indexes = self.random.permutation(size)[:nint(size * self.subsample)]
        indexes.sort()
        return np.asarray(Xb)[indexes], np.asarray(yb)[indexes]


class NeuralNet(nolearn.lasagne.NeuralNet):
    def __init__(self, layers, **kwargs):
        self.subsample = kwargs.pop('subsample', 1.0)
        self.batch_size = kwargs.pop('batch_size', 128)
        self.target_transforms = kwargs.pop('target_transforms', None)
        super(NeuralNet, self).__init__(layers, **kwargs)
        self.batch_iterator_train = BatchIterator(batch_size=self.batch_size, subsample=self.subsample, seed=42)

    def _check_good_input(self, X, y=None):
        if isinstance(X, dict):
            lengths = [len(X1) for X1 in X.values()]
            if len(set(lengths)) > 1:
                raise ValueError("Not all values of X are of equal length.")
            x_len = lengths[0]
        else:
            x_len = len(X)

        if y is not None:
            if len(y) != x_len:
                raise ValueError("X and y are not of equal length.")

        if self.regression and y is not None and y.ndim == 1:
            y = y.reshape(-1, 1)

        return X, y

    def fit(self, X, y, Xt=None, yt=None):
        X, y = self._check_good_input(X, y)
        if Xt is not None:
            Xt, yt = self._check_good_input(Xt, yt)

        if self.use_label_encoder:
            self.enc_ = LabelEncoder()
            y = self.enc_.fit_transform(y).astype(np.int32)
            self.classes_ = self.enc_.classes_
        self.initialize()

        try:
            self.train_loop(X, y, Xt, yt)
        except KeyboardInterrupt:
            pass
        return self

    def train_loop(self, X, y, Xt, yt):
        X_ = X.astype('float32')
        if self.regression:
            y_ = y.astype('float32')
        else:
            y_ = y.astype('int32')
        if Xt is not None:
            Xt_ = Xt.astype('float32')
            if self.regression:
                yt_ = yt.astype('float32')
            else:
                yt_ = yt.astype('int32')

        if self.target_transforms and self.regression:
            y_ = self.target_transforms[0](y_).astype('float32')
            yt_ = self.target_transforms[0](yt_).astype('float32')

        X_train, X_valid, y_train, y_valid = X_, Xt_, y_, yt_
        if X_valid is None or y_valid is None:
            X_valid = X[len(X):]
            y_valid = y[len(y):]

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_started = self.on_training_started
        if not isinstance(on_training_started, (list, tuple)):
            on_training_started = [on_training_started]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        epoch = 0
        best_valid_loss = (
            min([row['valid_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        best_train_loss = (
            min([row['train_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        for func in on_training_started:
            func(self, self.train_history_)

        num_epochs_past = len(self.train_history_)

        while epoch < self.max_epochs:
            epoch += 1

            train_losses = []
            valid_losses = []
            valid_accuracies = []
            custom_score = []

            t0 = time()

            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                batch_train_loss = self.apply_batch_func(
                    self.train_iter_, Xb, yb)
                train_losses.append(batch_train_loss)

            for Xb, yb in self.batch_iterator_test(X_valid, y_valid):
                batch_valid_loss, accuracy = self.apply_batch_func(
                    self.eval_iter_, Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)

                if self.custom_score:
                    y_prob = self.apply_batch_func(self.predict_iter_, Xb)
                    custom_score.append(self.custom_score[1](yb, y_prob))

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)
            if custom_score:
                avg_custom_score = np.mean(custom_score)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

            info = {
                'epoch': num_epochs_past + epoch,
                'train_loss': avg_train_loss,
                'train_loss_best': best_train_loss == avg_train_loss,
                'valid_loss': avg_valid_loss,
                'valid_loss_best': best_valid_loss == avg_valid_loss,
                'valid_accuracy': avg_valid_accuracy,
                'dur': time() - t0,
                }
            if self.custom_score:
                info[self.custom_score[0]] = avg_custom_score
            self.train_history_.append(info)

            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

        for func in on_training_finished:
            func(self, self.train_history_)

    @staticmethod
    def apply_batch_func(func, Xb, yb=None):
        if isinstance(Xb, dict):
            kwargs = dict(Xb)
            if yb is not None:
                kwargs['y'] = yb
            return func(**kwargs)
        else:
            return func(Xb) if yb is None else func(Xb, yb)

    def predict_proba(self, X):
        X_ = X.astype('float32')
        yp = super(NeuralNet, self).predict_proba(X_)
        return yp

    def predict(self, X):
        X_ = X.astype('float32')
        yp = super(NeuralNet, self).predict(X_)
        if self.target_transforms and self.regression:
            yp = self.target_transforms[1](yp)
        return yp


class AnnealVariable(object):
    def __init__(self, name, start=0.01, T=30):
        self.name = name
        self.start = start
        self.T = T
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            anneal = lambda x: self.start / ( 1 + x / self.T )
            self.ls = anneal(np.arange(float(nn.max_epochs)))
        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001, stop_epoch=None):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None
        self.stop_epoch = stop_epoch

    def __call__(self, nn, train_history):
        if self.stop_epoch is None:
            stop_epoch = nn.max_epochs
        else:
            stop_epoch = self.stop_epoch
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, stop_epoch)
        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


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


class SaveBest(object):
    def __init__(self):
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
        if current_epoch == nn.max_epochs:
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)


class SavePredictions(object):
    def __init__(self, epochs, filename_str, X):
        self.epochs = epochs
        self.filename_str = filename_str
        self.X = X

    def __call__(self, nn, train_history):
        current_epoch = train_history[-1]['epoch']
        if current_epoch in self.epochs:
            filename = self.filename_str.replace("_final_",str(current_epoch))
            print "Saving to filename:",filename
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError, e:
                if e.errno != errno.EEXIST:
                    raise
            pred = np.expm1(nn.predict( self.X ))
            with open(filename, 'w') as f:
                pred.astype(np.float32).tofile(f)


def make_name(dataset, params, string):
    name = ['%s'%dataset, 'nn']
    for layer in [layer[0] for layer in params['layers']]:
        if layer in ['input', 'output']: continue
        s = params.get('%s_p'%layer, None)
        if s: name.append('d%s'%s)
        s = params.get('%s_num_units'%layer, None)
        if s: name.append('%i'%s)
    s = params.get('subsample', None)
    if s: name.append('s%i'%s)
    name.append(string)
    return '-'.join(name)





