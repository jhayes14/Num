#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Chrissly31415 August 2013


from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator

from scipy.optimize import fmin, fmin_cobyla, minimize

from qsprLib import funcdict, save_sparse_csr, load_sparse_csr

import scipy as sp
import numpy as np
import pandas as pd
import pickle


class XModel:
    """
    class holds classifier plus train and test set for later use in ensemble building
    Wrapper for ensemble building
    """
    modelcount = 0

    def __init__(self, name, classifier, Xtrain, Xtest, ytrain=None, sample_weight=None, Xval=None, yval=None,
                 cutoff=None, class_names=None, cv_labels=None, bag_mode=False):
        self.name = name
        self.classifier = classifier
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.Xval = Xval
        self.yval = yval
        self.class_names = class_names
        self.cv_labels = cv_labels  # for special cv like leave pout
        self.bag_mode = bag_mode  # bagging i.e. cross-validated models for prediction

        if isinstance(Xtrain, sp.sparse.csr.csr_matrix) or isinstance(Xtrain, sp.sparse.csc.csc_matrix):
            self.sparse = True
        else:
            self.sparse = False
        self.sample_weight = sample_weight

        self.oob_preds = np.zeros((Xtrain.shape[0], 1))
        self.preds = np.zeros((Xtest.shape[0], 1))
        if Xval is not None:
            self.val_preds = np.zeros((Xval.shape[0], 1))

        if Xval is not None and sample_weight is not None:
            raise Exception("Holdout and sampleweight currently not supported!")
        self.cutoff = cutoff
        if cutoff is not None:
            self.use_proba = True

        XModel.modelcount += 1

    def summary(self):
        print ">>Name<<     :", self.name
        print "classifier   :", type(self.classifier)
        print "Train data   :", self.Xtrain.shape,
        print " type         :", type(self.Xtrain)
        print "Test data    :", self.Xtest.shape,
        print " type         :", type(self.Xtest)

        if self.Xval is not None:
            print "Valid. data   : ", self.Xval.shape
        if self.sample_weight is not None:
            print "sample_weight:", self.sample_weight.shape,
            print " type        :", type(self.sample_weight)

        if self.ytrain is not None:
            print "y <-  target :", self.ytrain.shape
        # print "sparse data  :" , self.sparse
        if self.cutoff is not None: print "proba cutoff  :", self.cutoff
        if self.class_names is not None: print "class names  :", self.class_names

        print "<predictions> : %6.3f" % (np.mean(self.preds)),
        print " Dim:", self.preds.shape
        print "<oob preds>   : %6.3f" % (np.mean(self.oob_preds)),
        print " Dim:", self.oob_preds.shape

    def __repr__(self):
        self.summary()

    # static function for saving
    @staticmethod
    def saveModel(xmodel, filename):
        if not hasattr(xmodel, 'xgboost_model'):
            # if not isinstance(xmodel,XgboostRegressor):
            pickle_out = open(filename.replace('.csv', ''), 'wb')
            pickle.dump(xmodel, pickle_out)
            pickle_out.close()

    @staticmethod
    def saveDataSet(xmodel, restoreOrder=True, basedir='./share/'):
        if xmodel.sparse:
            save_sparse_csr(basedir + "Xtrain_" + xmodel.name + "_sparse.csv", xmodel.Xtrain)
            save_sparse_csr(basedir + "Xtest_" + xmodel.name + "_sparse.csv", xmodel.Xtest)
        else:
            # Xtrain_ta = pd.concat([pd.DataFrame(self.cv_labels,columns=['tube_assembly_id'],index=self.Xtrain.index),self.Xtrain],axis=1)
            xmodel.Xtrain.to_csv(basedir + "Xtrain_" + xmodel.name + ".csv", index=False)
            # Xtest_ta = pd.concat([pd.DataFrame(_,columns=['tube_assembly_id'],index=Xtest.index),Xtest],axis=1)
            xmodel.Xtest.to_csv(basedir + "Xtest_" + xmodel.name + ".csv", index=False)

    @staticmethod
    def loadDataSet(xmodel, restoreOrder=True, basedir='./share/'):
        if xmodel.sparse:
            xmodel.Xtrain = load_sparse_csr(basedir + "Xtrain_" + xmodel.name + "_sparse.csv.npz")
            xmodel.Xtest = load_sparse_csr(basedir + "Xtest_" + xmodel.name + "_sparse.csv.npz")
        else:
            # Xtrain_ta = pd.concat([pd.DataFrame(self.cv_labels,columns=['tube_assembly_id'],index=self.Xtrain.index),self.Xtrain],axis=1)
            xmodel.Xtrain = pd.read_csv(basedir + "Xtrain_" + xmodel.name + ".csv")
            # Xtest_ta = pd.concat([pd.DataFrame(_,columns=['tube_assembly_id'],index=Xtest.index),Xtest],axis=1)
            xmodel.Xtest = pd.read_csv(basedir + "Xtest_" + xmodel.name + ".csv")

        return (xmodel.Xtrain, xmodel.Xtest)

    # static function for saving only the important parameters
    @staticmethod
    def saveCoreData(xmodel, filename):
        if not hasattr(xmodel, 'xgboost_model'):
            # reset not needed stuff
            xmodel.classifier = str(xmodel.classifier)
        xmodel.Xtrain = None
        xmodel.Xtest = None
        xmodel.sample_weight = None
        # keep only parameters and predictions
        pickle_out = open(filename.replace('.csv', ''), 'wb')
        pickle.dump(xmodel, pickle_out)
        pickle_out.close()

    # static function for loading
    @staticmethod
    def loadModel(filename):
        my_object_file = open(filename + '.pkl', 'rb')
        xmodel = pickle.load(my_object_file)
        my_object_file.close()
        return xmodel


class FeaturePredictor(BaseEstimator):
    """
    Yields single column(s) with Feature
    """

    def __init__(self, fname):
        self.fname = fname

    def fit(self, lX, ly=None, sample_weight=None):
        pass

    def predict(self, lX):
        return lX[self.fname].values


class NothingTransform(BaseEstimator):
    """
    Yields single column(s) with Feature
    """

    def fit(self, lX, ly=None, sample_weight=None):
        pass

    def fit_transform(self, lX, ly=None, sample_weight=None):
        return lX

    def transform(self, lX):
        return lX

    def predict(self, lX):
        pass


class ConstrainedLinearRegressor(BaseEstimator):
    """
        Constrained linear regression
        """

    def __init__(self, lowerbound=0, upperbound=1.0, n_classes=1, alpha=None, corr_penalty=None, normalize=False,
                 loss='rmse', greater_is_better=False):
        self.normalize = normalize
        self.greater_is_better = greater_is_better
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.n_classes = n_classes
        self.alpha = alpha
        self.loss = loss
        self.coef_ = 0
        self.corr_penalty = corr_penalty

    def fit(self, lX, ly, sample_weight=None):
        n_cols = lX.shape[1]
        x0 = np.ones((n_cols, 1))
        constr_lb = [lambda x, z=i: x[z] - self.lowerbound for i in range(n_cols)]
        constr_ub = [lambda x, z=i: self.upperbound - x[z] for i in range(n_cols)]
        constr = constr_lb + constr_ub
        # constr=constr_lb
        self.coef_ = fmin_cobyla(self.fopt, x0, constr, args=(lX, ly), consargs=(), rhoend=1e-10, maxfun=10000, disp=0)
        # coef_ = minimize(fopt, x0,method='COBYLA',constraints=self.constr)
        # normalize coefficient
        if self.normalize:
            self.coef_ = self.coef_ / np.sum(self.coef_)
            # print "Normalizing coefficients:",self.coef_

        if np.isnan(np.sum(self.coef_)):
            print "We have NaN here..."

    def predict(self, lX):
        ypred = self.blend_mult(lX, self.coef_, self.n_classes)
        return ypred.flatten()

    def predict_proba(self, lX):
        print "proba not implemented yet..."
        pass

    def fopt(self, params, X, y):
        # nxm  * m*1 ->n*1
        ypred = self.blend_mult(X, params, self.n_classes)
        score = funcdict[self.loss](y, ypred)
        # if not use_proba: ypred = np.round(ypred).astype(int)
        # regularization
        if self.alpha is not None:
            # cc = np.corrcoef(X.values,rowvar=0)
            # cc = np.power(cc,2)
            # cc = self.corr_penalty * np.mean(cc)
            l2 = self.alpha * np.sum(np.power(params, 2))
            # l1 = self.alpha *np.sum(np.absolute(params))
            score = score + l2
        # print "score: %6.2f alpha: %6.2f cc: %r l2: %6.3f"%(score,self.alpha,cc,l2)
        # raw_input()
        if self.greater_is_better: score = -1 * score
        return score

    def blend_mult(self, X, params, n_classes=None):
        if n_classes < 2:
            return np.dot(X, params)
            #    else:
            #        return multiclass_mult(X,params,n_classes)
