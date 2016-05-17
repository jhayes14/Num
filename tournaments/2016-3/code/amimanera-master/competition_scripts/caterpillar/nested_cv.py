#!/usr/bin/python 
# coding: utf-8

'''
Created on 26 Aug 2015

@author: loschen
'''

import numpy as np
import pandas as pd


from sklearn.cross_validation import KFold, PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,Normalizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from FullModel import *
from qsprLib import *
from nslearn import *

from cater import *
import time

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

def root_mean_squared_logarithmic_error(true, pred):
     return np.sqrt( mean_squared_error( np.log1p(true), np.log1p(pred) ) )

preprocess = make_pipeline(
                StandardScaler(),
)

def loadDataSet_old(model):
    """
    parallel oob creation
    """
    basedir="./data/"
    xmodel = XModel.loadModel(basedir+model)
    Xtrain,Xtest = XModel.loadDataSet(xmodel)
    ta = pd.read_csv("./data/train_set.csv", usecols=['tube_assembly_id']).values.ravel()
    y = pd.read_csv("./data/train_set.csv", usecols=['cost']).values.ravel()
    print "model: %-20s %20r %20r "%(xmodel.name,Xtrain.shape,Xtest.shape)
    return Xtrain,Xtest,ta,y.ravel()

def loadDataSet(df_name):
    Xtrain = pd.read_csv("./share/Xtrain_"+df_name+".csv")
    Xtest = pd.read_csv("./share/Xtest_"+df_name+".csv")

    ta = pd.read_csv("./data/train_set.csv", usecols=['tube_assembly_id']).values.ravel()
    y = pd.read_csv("./data/train_set.csv", usecols=['cost']).values.ravel()
    return Xtrain,Xtest,ta,y

if __name__=="__main__":
    random_state = 42
    cwd = os.getcwd()
    dataset_model = "nn8_br20"
    epoch_save_range = range(2, 10, 2)
    #reg_orig = nnet_cater3
    reg_orig = nnet_BN1
    #reg_orig = BaggingRegressor(base_estimator=nnet_BN1,n_estimators=3,n_jobs=1,verbose=2,random_state=None,max_samples=1.0,max_features=1.0,bootstrap=False)

    netconfig = "nn10_BN1"
    param_list=[5.0*1E-4]
    orig_name = "{0}-{1}".format(dataset_model, netconfig)

    #target = target.ravel()
    stage1 = os.path.join('./stage1')
    refit_train_val_dir = 'refit_train_val'
    refit_train_dir = 'refit_train'

    test,train,target,_,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=False,loadBN='somefolder')
    #train,test,ta,target = loadDataSet(dataset_model)
    #train,test = scaleData(train,test)
    print target


    print train.shape
    print test.shape
    print ta.shape


    t0 = time.time()
    n_folds = 3
    do_folds = 3

    kfold_outer = KLabelFold(pd.Series(ta), n_folds, shuffle=True, random_state=42)
    for fold_outer, val_mask in enumerate(kfold_outer):
        print ''.join(['#']*120)
        print 'fold_outer ', fold_outer
        fold_outer_dir = 'fold_outer-%s'%(fold_outer)
        X_train_val = train.iloc[val_mask[0]]
        y_train_val = target[val_mask[0]]
        X_val = train.iloc[val_mask[1]]
        y_val = target[val_mask[1]]

        print "X_train_val:",X_train_val.shape
        print "X_val:",X_val.shape
        kfold_inner = KLabelInnerFold(kfold_outer, fold_outer)
        t0_outer = time.time()

        for iparam, params in enumerate(param_list):
            print ''.join(['-']*90)
            name = orig_name+'_param%i'%(iparam)
            for inner_folds_, mask in kfold_inner:
                train_folds, fold_inner = inner_folds_
                print ''.join(['-']*60)
                print 'fold_inner ', fold_inner, 'train_folds_inner ', train_folds
                fold_inner_dir = 'fold_inner-%s'%(fold_inner)

                X_train = train.iloc[mask[0]]
                print "Xtrain:",X_train.shape
                y_train = target[mask[0]]

                X_test = train.iloc[mask[1]]
                print "Xtest:",X_test.shape
                y_test = target[mask[1]]

                X_train = preprocess.fit_transform( X_train )
                X_test = preprocess.transform( X_test )
                X_val_ = preprocess.transform( X_val )
                #test_ = preprocess.transform( test )

                reg = clone(reg_orig)
                #reg.set_params({'objective_alpha':params,'maxp_epoch':20})
                directory = os.path.join(cwd, 'stage2', name,'_final_', fold_outer_dir, fold_inner_dir)
                #reg.__setattr__('max_epochs',20)
                reg.__setattr__('on_epoch_finished',[
                AdjustVariable('update_learning_rate', start=0.0005, stop=0.0000001),
                SavePredictions(epoch_save_range,os.path.join(directory, 'y_pred_test_float32'),X_val_),
                SavePredictions(epoch_save_range,os.path.join(directory, 'y_pred_val_float32'),X_val_)])
                reg.fit(X_train,np.log1p(y_train))

                y_pred_test = np.expm1( reg.predict( X_test) )
                print 'RMSLE(Xtest) = %.5f'%( root_mean_squared_logarithmic_error( y_test, y_pred_test ) )

                os.makedirs(directory)
                print "Saving prediction for (inner) X_test.."
                with open(os.path.join(directory, 'y_pred_test_float32'), 'w') as f:
                        y_pred_test.astype(np.float32).tofile(f)

                y_pred_val = np.expm1( reg.predict( X_val_ ))
                with open(os.path.join(directory, 'y_pred_val_float32'), 'w') as f:
                        y_pred_val.astype(np.float32).tofile(f)


            print ''.join(['#']*90)
            print 'Inner loop finished - refit on X_train_val set'
            X_train_val_ = preprocess.fit_transform( X_train_val )
            X_val_ = preprocess.transform( X_val )

            reg = clone(reg_orig)
            directory = os.path.join(cwd, 'stage2', name,'_final_', fold_outer_dir, refit_train_val_dir)
            reg.__setattr__('on_epoch_finished',[
                AdjustVariable('update_learning_rate', start=0.0005, stop=0.0000001),
                SavePredictions(epoch_save_range,os.path.join(directory, 'y_pred_test_float32'),X_val_),
                SavePredictions(epoch_save_range,os.path.join(directory, 'y_pred_val_float32'),X_val_)])
            reg.fit(X_train_val_, np.log1p(y_train_val))
            print 'RMSLE(X_val) = %.5f'%(root_mean_squared_logarithmic_error( y_val, y_pred_val ) )


            os.makedirs(directory)
            with open(os.path.join(directory, 'y_pred_val_float32'), 'w') as f:
                    y_pred_val.astype(np.float32).tofile(f)

            if fold_outer == 0:
                print ''.join(['-']*90)
                print 'refit on ALL train data'
                train_ = preprocess.fit_transform(train)
                test_ = preprocess.transform(test)
                reg = clone(reg_orig)
                directory = os.path.join(cwd, 'stage2', name,'_final_', refit_train_dir)
                reg.__setattr__('on_epoch_finished',[
                AdjustVariable('update_learning_rate', start=0.0005, stop=0.0000001),
                SavePredictions(epoch_save_range,os.path.join(directory, 'y_pred_test_float32'),X_val_),
                SavePredictions(epoch_save_range,os.path.join(directory, 'y_pred_val_float32'),X_val_)])

                reg.fit(train_, np.log1p(target))
                y_pred_sub = np.expm1( reg.predict( test_) )

                os.makedirs(directory)
                with open(os.path.join(directory, 'y_pred_sub_float32'), 'w') as f:
                        y_pred_sub.astype(np.float32).tofile(f)


            t = (time.time() - t0_outer)/(60*60)
            print 'finished fold_outer %s  at time %.2fs'%( fold_outer,  t)
    
    t = (time.time() - t0)/(60*60)   
    print 'finished fold %s of %s at time %.2fh'%(fold_outer+1, n_folds, t)
        