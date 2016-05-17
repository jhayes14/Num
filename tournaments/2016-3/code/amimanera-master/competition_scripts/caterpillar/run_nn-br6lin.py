import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import copy
from sklearn.cross_validation import KFold, PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import sklearn.utils
import sklearn
from preprocessing import ScaleContinuousOnly, PCAContinuousOnly
import theano
import time
from nslearn import *

from lasagne_tools import nnet_cater3

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def root_mean_squared_logarithmic_error(true, pred):
     return np.sqrt( mean_squared_error( np.log1p(true), np.log1p(pred) ) )

random_state = 42


stage1 = os.path.join(os.path.expanduser('~'), 'kaggle/Caterpillar/stage1')
data_path = os.path.join(os.path.expanduser('~'), 'kaggle/Caterpillar/data')

dataset = 'br6lin'

train = pd.read_csv(os.path.join(stage1, dataset, 'training.csv'))
target = train.cost
train.drop(['cost'], axis=1, inplace=True)
tube_assembly_ids = pd.read_csv(os.path.join(data_path, 'train_set.csv'), usecols=['tube_assembly_id'])
train['tube_assembly_id'] = tube_assembly_ids

test = pd.read_csv(os.path.join(stage1, dataset, 'testing.csv'))
test.drop(['cost'], axis=1, inplace=True)


preprocess = make_pipeline(
                ScaleContinuousOnly(),
#                StandardScaler(),
)

cwd = os.getcwd()
refit_train_val_dir = 'refit_train_val'
refit_train_dir = 'refit_train'

#epoch_save_range = range(30, 101, 5)
epoch_save_range = range(10, 11, 5)

t0 = time.time()
n_folds = 3
do_folds = 3

def KLabelFold(labels, n_folds=3, shuffle=False, random_state=None):
    kfold = KFold(labels.nunique(), n_folds=n_folds, shuffle=shuffle, random_state=random_state)
    unique_labels = labels.unique()
    return PredefinedSplit(pd.concat(map(lambda (i, x): labels.isin(x)*i,
                                                        enumerate([unique_labels[mask[1]] for mask in kfold])
                                        ), axis=1).sum(axis=1))

def KLabelInnerFold(kfold_outer, fold_outer):
    indicies = np.arange(kfold_outer.n)
    for test_fold in kfold_outer.unique_folds[kfold_outer.unique_folds != fold_outer]:
        if test_fold == fold_outer:
            continue
        train_folds = list(kfold_outer.unique_folds[(kfold_outer.unique_folds != fold_outer) & (kfold_outer.unique_folds != test_fold)])
        yield (train_folds, test_fold),\
              (indicies[pd.Series(kfold_outer.test_fold).isin(train_folds).values], indicies[kfold_outer.test_fold == test_fold])

kfold_outer = KLabelFold(train.tube_assembly_id, n_folds, shuffle=True, random_state=42)
for fold_outer, val_mask in enumerate(kfold_outer):
    print ''.join(['-']*120)
    print 'fold_outer ', fold_outer
    fold_outer_dir = 'fold_outer-%s'%(fold_outer)
    X_train_val = train.iloc[val_mask[0]].drop(['tube_assembly_id'], axis=1)
    y_train_val = target.iloc[val_mask[0]]
    X_val = train.iloc[val_mask[1]].drop(['tube_assembly_id'], axis=1)
    y_val = target.iloc[val_mask[1]]

    t0_outer = time.time()
    for iparam, params in enumerate([1]):
        print ''.join(['-']*90)
#        print 'model %s of %s '%(iparam+1, len(param_list))
#        name = make_name(dataset, nn_params, 'sco')
#        if os.path.exists(os.path.join(cwd, 'stage2', name, str(epoch_save_range[-1]), fold_outer_dir)):
#            print name, ' exists, skipping parameter set'
#            continue
        name = 'br6lin-nn'
        print name

        kfold_inner = KLabelInnerFold(kfold_outer, fold_outer)
        for inner_folds_, mask in kfold_inner:
            train_folds, fold_inner = inner_folds_
            print ''.join(['-']*60)
            print 'fold_inner ', fold_inner, 'train_folds_inner ', train_folds
            fold_inner_dir = 'fold_inner-%s'%(fold_inner)

            X_train = train.iloc[mask[0]].drop(['tube_assembly_id'], axis=1)
            y_train = target.iloc[mask[0]].astype('float32').values.reshape(-1, 1)

            X_test = train.iloc[mask[1]].drop(['tube_assembly_id'], axis=1)
            y_test = target.iloc[mask[1]].astype('float32').values.reshape(-1, 1)

            X_train_ = preprocess.fit_transform( X_train ).astype('float32').values
            X_test_ = preprocess.transform( X_test ).astype('float32').values
            X_val_ = preprocess.transform( X_val ).astype('float32').values
            test_ = preprocess.transform( test ).astype('float32').values
            X_train_, y_train_ = sklearn.utils.shuffle(X_train_, y_train, random_state=random_state)

            directory = os.path.join(cwd, 'stage2', name, '%i'%(iparam), fold_outer_dir, fold_inner_dir)
            reg = sklearn.clone(nnet_cater3)
#            reg.fit(X_train_, y_train_, X_test_, y_test)
#            reg.fit(X_train_, np.log1p(y_train_), X_test_, np.log1p(y_test))
            reg.fit(X_train_, np.log1p(y_train_))

            y_pred_test = np.expm1( reg.predict( X_test_ ) )
            print 'RMSLE (%5s)= %.5f'%( epoch_save_range[-1], root_mean_squared_logarithmic_error( y_test, y_pred_test ) )

        print ''.join(['-']*90)
        print 'refit on train_val set'
        X_train_val_ = preprocess.fit_transform( X_train_val )
        X_val_ = preprocess.transform( X_val )
        X_train_val_, y_train_val_ = sklearn.utils.shuffle(X_train_val_, y_train_val, random_state=random_state)

        directory = os.path.join(cwd, 'stage2', name, '%i', fold_outer_dir, refit_train_val_dir)
#        os.makedirs(directory)
        nn_params = {}
        nn_params.update(core_params)
        nn_params.update(params)
        nn_params.update(dict(input_shape=(None, X_train_val_.shape[1])))
        nn_params['on_epoch_finished'].extend([
                            SavePredictions(epoch_save_range,
                                            os.path.join(directory, 'y_pred_val_float32'),
                                            X_val_)])
        reg = NeuralNet(**nn_params)
        reg.fit(X_train_val_, y_train_val_, X_val_, y_val)
        y_pred_val = reg.predict( X_val_ )
        print 'RMSLE (%5s)= %.5f'%( epoch_save_range[-1], root_mean_squared_logarithmic_error( y_val, y_pred_val ) )

        if fold_outer == 0:
            print ''.join(['-']*90)
            print 'refit on all train data'
            train_ = preprocess.fit_transform( train.drop(['tube_assembly_id'], axis=1) )
            test_ = preprocess.transform( test )
            train_, target_ = sklearn.utils.shuffle(train_, target, random_state=random_state)

            directory = os.path.join(cwd, 'stage2', name, '%i', refit_train_dir)
#            os.makedirs(directory)
            nn_params = {}
            nn_params.update(core_params)
            nn_params.update(params)
            nn_params.update(dict(input_shape=(None, train_.shape[1])))
            nn_params['on_epoch_finished'].extend([
                                SavePredictions(epoch_save_range,
                                                os.path.join(directory, 'y_pred_sub_float32'),
                                                test_)])
            reg = NeuralNet(**nn_params)
            reg.fit(train_, target_)

        t = (time.time() - t0_outer)/(60*60)
        eta_outer = (len(param_list)-(iparam+1))*t
        print 'finished fold_outer %s for model %s of %s at time %.2fh, ETA (outer) %.2fh'%(
                        fold_outer, iparam+1, len(param_list), t, eta_outer)

    t = (time.time() - t0)/(60*60)
    eta = (n_folds-(fold_outer+1))*t
    print 'finished fold %s of %s at time %.2fh, ETA %.2fh'%(fold_outer+1, n_folds, t, eta)

    if do_folds-1 == fold_outer:
        break
