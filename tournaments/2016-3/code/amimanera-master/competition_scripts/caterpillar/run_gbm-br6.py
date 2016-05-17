import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import copy
from sklearn.cross_validation import KFold, PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import sklearn.utils
import sklearn
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def root_mean_squared_logarithmic_error(true, pred):
     return np.sqrt( mean_squared_error( np.log1p(true), np.log1p(pred) ) )

random_state = 42


stage1 = os.path.join(os.path.expanduser('~'), 'kaggle/Caterpillar/stage1')
data_path = os.path.join(os.path.expanduser('~'), 'kaggle/Caterpillar/data')

dataset = 'br6'

train = pd.read_csv(os.path.join(stage1, dataset, 'training.csv'))
target = train.cost
train.drop(['cost'], axis=1, inplace=True)
tube_assembly_ids = pd.read_csv(os.path.join(data_path, 'train_set.csv'), usecols=['tube_assembly_id'])
train['tube_assembly_id'] = tube_assembly_ids

test = pd.read_csv(os.path.join(stage1, dataset, 'testing.csv'))
test.drop(['cost'], axis=1, inplace=True)


preprocess = make_pipeline(
                DoNothingTransform(),
)

cwd = os.getcwd()
refit_train_val_dir = 'refit_train_val'
refit_train_dir = 'refit_train'

tree_range = range(100, 2100, 100)

param_list = [
    dict(learning_rate=0.1, max_depth=25, max_features=1.0, subsample=1.0, min_samples_leaf=2),
    dict(learning_rate=0.1, max_depth=30, max_features=1.0, subsample=1.0, min_samples_leaf=2),
    dict(learning_rate=0.1, max_depth=35, max_features=1.0, subsample=1.0, min_samples_leaf=2),
    dict(learning_rate=0.1, max_depth=40, max_features=1.0, subsample=1.0, min_samples_leaf=2),
    dict(learning_rate=0.1, max_depth=35, max_features=1.0, subsample=0.9, min_samples_leaf=2),
    dict(learning_rate=0.1, max_depth=35, max_features=0.9, subsample=1.0, min_samples_leaf=2),
]

#tree_range = tree_range[:1]
#param_list = param_list[:2]

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
    for iparam, params in enumerate(param_list):
        print ''.join(['-']*90)
        print 'model %s of %s '%(iparam+1, len(param_list))
        gbm_params = dict(
                    n_estimators=tree_range[-1],
                    verbose=False,
                    random_state=42,
        )
        gbm_params.update(params)
        name = '%s-gbm-l%.2f-d%i-f%.1f-s%.1f-w%i'%(dataset,
                                              gbm_params['learning_rate'],
                                              gbm_params['max_depth'],
                                              gbm_params['max_features'],
                                              gbm_params['subsample'],
                                              gbm_params['min_samples_leaf'])
        if os.path.exists(os.path.join(cwd, 'stage2', name, str(tree_range[-1]), fold_outer_dir)):
            print name, ' exists, skipping parameter set'
            continue
        print name

        kfold_inner = KLabelInnerFold(kfold_outer, fold_outer)
        for inner_folds_, mask in kfold_inner:
            train_folds, fold_inner = inner_folds_
            print ''.join(['-']*60)
            print 'fold_inner ', fold_inner, 'train_folds_inner ', train_folds
            fold_inner_dir = 'fold_inner-%s'%(fold_inner)

            X_train = train.iloc[mask[0]].drop(['tube_assembly_id'], axis=1)
            y_train = target.iloc[mask[0]]

            X_test = train.iloc[mask[1]].drop(['tube_assembly_id'], axis=1)
            y_test = target.iloc[mask[1]]

            X_train = preprocess.fit_transform( X_train )
            X_test = preprocess.transform( X_test )
            X_val_ = preprocess.transform( X_val )
            test_ = preprocess.transform( test )
#            X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=fold_seed[seed])

            reg = GBMRegressor(**gbm_params)
            reg.fit(X_train, np.log1p(y_train))#, X_test, np.log1p(y_test))

            for ntree in tree_range:
                directory = os.path.join(cwd, 'stage2', name, '%i'%ntree, fold_outer_dir, fold_inner_dir)
                os.makedirs(directory)
                y_pred_test = np.expm1( reg.predict( X_test, ntree_limit=ntree ) )
                print 'RMSLE (%5s)= %.5f'%( ntree, root_mean_squared_logarithmic_error( y_test, y_pred_test ) )
                with open(os.path.join(directory, 'y_pred_test_float32'), 'w') as f:
                    y_pred_test.astype(np.float32).tofile(f)
                y_pred_val = np.expm1( reg.predict( X_val_, ntree_limit=ntree ) )
                with open(os.path.join(directory, 'y_pred_val_float32'), 'w') as f:
                    y_pred_val.astype(np.float32).tofile(f)

        print ''.join(['-']*90)
        print 'refit on train_val set'
        X_train_val = preprocess.fit_transform( X_train_val )
        X_val = preprocess.transform( X_val )
        #X_train_val, y_train_val = sklearn.utils.shuffle(X_train_val, y_train_val, random_state=random_state)
        params = gbm_params.copy()
        reg = GBMRegressor(**params)
        reg.fit(X_train_val, np.log1p(y_train_val))#, X_val, np.log1p(y_val))
        for ntree in tree_range:
            directory = os.path.join(cwd, 'stage2', name, '%i'%ntree, fold_outer_dir, refit_train_val_dir)
            os.makedirs(directory)
            y_pred_val = np.expm1( reg.predict( X_val, ntree_limit=ntree ) )
            print 'RMSLE (%5s)= %.5f'%( ntree, root_mean_squared_logarithmic_error( y_val, y_pred_val ) )
            with open(os.path.join(directory, 'y_pred_val_float32'), 'w') as f:
                y_pred_val.astype(np.float32).tofile(f)

        if fold_outer == 0:
            print ''.join(['-']*90)
            print 'refit on all train data'
            train_ = preprocess.fit_transform( train.drop(['tube_assembly_id'], axis=1) )
            test_ = preprocess.transform( test )
            #train, target = sklearn.utils.shuffle(train, target, random_state=random_state)
            # need to figure out final number of epochs from training data
            params = gbm_params.copy()
            reg = GBMRegressor(**params)
            reg.fit(train_, np.log1p(target))
            for ntree in tree_range:
                directory = os.path.join(cwd, 'stage2', name, '%i'%ntree, refit_train_dir)
                os.makedirs(directory)
                y_pred_sub = np.expm1( reg.predict( test_, ntree_limit=ntree ) )
                with open(os.path.join(directory, 'y_pred_sub_float32'), 'w') as f:
                    y_pred_sub.astype(np.float32).tofile(f)

        t = (time.time() - t0_outer)/(60*60)
        eta_outer = (len(param_list)-(iparam+1))*t
        print 'finished fold_outer %s for model %s of %s at time %.2fh, ETA (outer) %.2fh'%(
                        fold_outer, iparam+1, len(param_list), t, eta_outer)

    t = (time.time() - t0)/(60*60)
    eta = (n_folds-(fold_outer+1))*t
    print 'finished fold %s of %s at time %.2fh, ETA %.2fh'%(fold_outer+1, n_folds, t, eta)

    if do_folds-1 == fold_outer:
        break
