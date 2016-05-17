#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd

# add path of xgboost python module
sys.path.append('/home/loschen/programs/xgboost/wrapper')
import xgboost as xgb

from sklearn.base import BaseEstimator
from sklearn import preprocessing

import qsprLib


class XgboostClassifier(BaseEstimator):
    """
    xgboost<-->sklearn interface
    Chrissly31415 July 2014
    xgboost: https://github.com/tqchen/xgboost
    sklearn: http://scikit-learn.org/stable/index.html
    based on the kaggle forum thread: https://www.kaggle.com/c/higgs-boson/forums/t/8184/public-starting-guide-to-get-above-3-60-ams-score/44691#post44691
    """

    def __init__(self, n_estimators=120, learning_rate=0.3, max_depth=6, subsample=1.0, min_child_weight=1,
                 colsample_bytree=1.0, gamma=0, objective='binary:logistic', eval_metric='auc', booster='gbtree',
                 n_jobs=1, cutoff=0.50, NA=-999.0, alpha_L1=0, lambda_L2=0, silent=1, eval_size=0.0):
        """
        Constructor
        Parameters: https://github.com/dmlc/xgboost/blob/d3af4e138f7cfa5b60a426f1468908f928092198/doc/parameter.md
        """
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.cutoff = cutoff
        self.NA = NA
        self.booster = booster
        self.objective = objective
        self.eval_metric = eval_metric
        self.subsample = subsample
        self.n_jobs = n_jobs
        self.silent = silent
        self.alpha_L1 = alpha_L1
        self.lambda_L2 = lambda_L2
        self.eval_size = eval_size

        self.isRegressor = False
        self.classes_ = -1
        self.xgboost_model = None
        self.encoder = None

    # self.param['scale_pos_weight'] = 1.0 #scaling can be done also externally| for AMS metric

    def fit(self, lX, ly, sample_weight=None):
        # avoid problems with pandas dataframes and DMatrix
        if isinstance(lX, pd.DataFrame): lX = lX.values
        if isinstance(ly, pd.Series) or isinstance(ly, pd.DataFrame): ly = ly.values


        if not self.isRegressor:
            #print "Encoding:"
            #print ly
            #print ly.shape
            self.classes_ = np.unique(ly)
            self.encoder = preprocessing.LabelEncoder()
            ly = self.encoder.fit_transform(ly)
            #print "After:"

        # if sample_weight is not None:
        #

        # early stopping!!
        if self.eval_size > 0.0:
            n_test = int(self.eval_size * lX.shape[0])
            idx_test = np.random.choice(xrange(lX.shape[0]), n_test, False)
            idx_train = [x for x in xrange(lX.shape[0]) if x not in idx_test]
            Xeval = lX[idx_test, :]
            yeval = ly[idx_test]
            lX = lX[idx_train, :]
            ly = ly[idx_train]
            deval = xgb.DMatrix(Xeval, label=yeval)

        ly = ly.reshape((ly.shape[0], -1))
        # xgmat = xgb.DMatrix(X, label=y, missing=self.NA, weight=sample_weight)#NA ??
        if sample_weight is not None:
            if isinstance(ly, pd.DataFrame): sample_weight = sample_weight.values
            dtrain = xgb.DMatrix(lX, label=ly, missing=self.NA, weight=sample_weight)
        else:
            dtrain = xgb.DMatrix(lX, label=ly, missing=self.NA)  # NA=0 as regulariziation->gives rubbish

        # set up parameters
        param = {}
        param['objective'] = self.objective  # 'binary:logitraw', 'binary:logistic', 'multi:softprob'
        param['eval_metric'] = self.eval_metric  # 'auc','mlogloss'
        #feval = None
        param['booster'] = self.booster  # gblinear
        param['subsample'] = self.subsample
        param['min_child_weight'] = self.min_child_weight
        param['colsample_bytree'] = self.colsample_bytree
        param['gamma'] = self.gamma
        param['bst:eta'] = self.learning_rate
        param['bst:max_depth'] = self.max_depth
        param['nthread'] = self.n_jobs
        param['silent'] = self.silent
        #if not self.isRegressor: param['num_class'] = np.unique(ly).shape[0]
        #print "num_class:",np.unique(ly).shape[0]
        param['alpha'] = self.alpha_L1
        param['lambda'] = self.lambda_L2

        plst = param.items()
        # watchlist = [ (dtrain,'train') ]
        if self.eval_size > 1E-15:
            watchlist = [(dtrain, 'train'), (deval, 'eval')]
            self.xgboost_model = xgb.train(plst, dtrain, num_boost_round=self.n_estimators, evals=watchlist,
                                           early_stopping_rounds=None, obj=None)
        else:
            watchlist = [(dtrain, 'train')]
            # self.xgboost_model = xgb.train(plst, dtrain, num_boost_round=self.n_estimators,evals=watchlist)
            self.xgboost_model = xgb.train(plst, dtrain, num_boost_round=self.n_estimators)

    def predict(self, lX):
        ly = self.predict_proba(lX)
        if 'multi:softprob' in self.objective:
            ly = np.argmax(ly)
        if not self.isRegressor:
            return self.encoder.inverse_transform(ly.astype(int))
        else:
            return ly

    def predict_proba(self, lX):
        # avoid problems with pandas dataframes and DMatrix
        if isinstance(lX, pd.DataFrame): lX = lX.values
        xgmat_test = xgb.DMatrix(lX, missing=self.NA)
        ly = self.xgboost_model.predict(xgmat_test)
        ly = np.column_stack((1.0 - ly,ly))
        return ly

    def get_fscore(self):
        return self.xgboost_model.get_fscore()


class XgboostRegressor(XgboostClassifier):
    def __init__(self, n_estimators=120, learning_rate=0.3, max_depth=6, subsample=1.0, min_child_weight=1,
                 colsample_bytree=1.0, gamma=0, objective='binary:logistic', eval_metric='auc', booster='gbtree',
                 n_jobs=1, cutoff=0.50, NA=-999.0, alpha_L1=0, lambda_L2=0, silent=1, eval_size=0.0):
        super(XgboostRegressor, self).__init__(n_estimators=n_estimators, learning_rate=learning_rate,
                                               max_depth=max_depth, subsample=subsample,
                                               min_child_weight=min_child_weight, colsample_bytree=colsample_bytree,
                                               gamma=gamma, objective=objective, eval_metric=eval_metric,
                                               booster=booster, n_jobs=n_jobs, cutoff=cutoff, NA=NA, alpha_L1=alpha_L1,
                                               lambda_L2=lambda_L2, silent=silent, eval_size=eval_size)
        self.isRegressor = True


## softmax
def softmax(score):
    print score
    print type(score)
    score = np.asarray(score, dtype=float)
    print score.shape
    score = np.exp(score - np.max(score))
    score /= np.sum(score, axis=1)[:, np.newaxis]
    return score


# evalerror is your customized evaluation function to 
# 1) decode the class probability 
# 2) compute quadratic weighted kappa
def evalerror_old(preds, dtrain):
    ## label are in [0,1,2,3] as required by XGBoost for multi-classification
    labels = dtrain.get_label() + 1
    pred_labels = preds + 1
    kappa = qsprLib.quadratic_weighted_kappa(labels, pred_labels)
    return 'kappa', kappa


#def evalerror(preds, dtrain):
#    ## label are in [0,1,2,3] as required by XGBoost for multi-classification
#    labels = dtrain.get_label()
#    score = qsprLib.root_mean_squared_error(preds, labels)
#    return 'rmse(mod)', score

def rmspe_xg(preds, dtrain):
    dtrain = np.expm1(dtrain.get_label())
    preds = np.expm1(preds)
    return "rmspe", qsprLib.root_mean_squared_percentage_error(dtrain,preds)


# user define objective function, given prediction, return gradient and second order gradient
# this is loglikelihood loss
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


if __name__ == "__main__":
    """
    test function
    """
    Xtrain, ytrain, Xtest, wtrain = higgs.prepareDatasets(1000)
    model = XgboostClassifier()
    model.fit(Xtrain, ytrain, wtrain)
    y = model.predict(Xtest)
