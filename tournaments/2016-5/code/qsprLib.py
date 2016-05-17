#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code fragment collection for QPSR. Using sklearn, pandas and numpy
"""
from sys import stdout
from time import time
import itertools
from random import choice
import numpy as np
import numpy.random as nprnd
import pandas as pd
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab as pl
# import tools
from sklearn.utils import shuffle
from sklearn import cross_validation, grid_search
from sklearn.cross_validation import StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit, train_test_split, \
    LeavePLabelOut
from sklearn.metrics import roc_auc_score, classification_report, make_scorer, f1_score, precision_score, \
    mean_squared_error, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif, f_regression, \
    GenericUnivariateSelect, VarianceThreshold
from sklearn.learning_curve import learning_curve
# import models
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.cluster import k_means
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression, SGDClassifier, Perceptron, \
    SGDRegressor, RidgeClassifier, LinearRegression, Ridge, BayesianRidge, ElasticNet, RidgeCV, LassoLarsCV, Lasso, \
    LassoCV, LassoLars, LarsCV, ElasticNetCV
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.decomposition import TruncatedSVD, PCA, RandomizedPCA, FastICA, MiniBatchSparsePCA, SparseCoder, \
    DictionaryLearning, MiniBatchDictionaryLearning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, ExtraTreesRegressor, GradientBoostingRegressor, BaggingRegressor, BaggingClassifier, \
    RandomForestRegressor, RandomTreesEmbedding
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
# from sklearn.calibration import CalibratedClassifierCV

#from xgboost_sklearn import *
import seaborn as sns


#####Check List
# Regression / Classification
# What kind of metric?
# Regression: gaussian distribution? Should we log transform data?
# Classification: Class imbalance?
# Multi-Classification: e.g. with kappa / ordered classes: shall we cast it into regression?
# Is the dataset ordered by an meany?  Does is contain a timeseries? Is the dataset grouped?
# How will it influence the CV procedure?
# Should we shuffle before training (e.g. important when using Neural Net wih mini-batch gradient training!)
# Size of dataset: Can we use a holdout dataset or can we even make a nested CV?
# Features: numerical, binary, categorical, ordered categorical?
# Interactive analysis of each feature, which can be made gaussian
# Was there anything similar on Kaggle!!!


def showCorrelations(X, steps=3):
    n = X.shape[1]
    idx_a = 0
    idx_b = n / steps

    for i in xrange(steps):
        print "a:%4d b: %4d" % (idx_a, idx_b)
        d = X.iloc[:, idx_a:idx_b]
        corr = d.corr()
        print corr
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
        #		square=True, xticklabels=2, yticklabels=2,
        #		linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

        sns.heatmap(corr, cmap=cmap, ax=ax)

        sns.set()
        # sns.pairplot(d)
        d.hist(bins=50)
        plt.show()
        idx_a = idx_b
        idx_b = idx_b + n / steps


def showAVGCorrelations(X_all, X_test=None):
    print "Showing correlated columns"
    c = X_all.corr().abs()

    corcols = {}
    for row in range(len(c.index)):
        av1 = c.iloc[row, :].mean()  # row mean
        print "model: %20s mean correl: %5.3f" % (c.index[row], av1)

"""X_0 = '../../../numerai_datasets_new/numerai_training_data.csv'
Y_0 = '../../../numerai_datasets_new/numerai_tournament_data.csv'
X = pd.read_csv( X_0 )
Y = pd.read_csv( Y_0 )
X = X.drop( 'target', axis = 1 )

Y = Y.drop( 't_id', axis = 1 )
#print X.columns
X_all = pd.concat([Y,X])

showAVGCorrelations(X_all)"""

def removeCorrelations(X_all, X_test=None, X_valid=None, threshhold=0.99):
    """
    Remove correlations, we could improve it by only removing the variable frmo two showing the highest correlations with others
    """
    print "Removing correlated columns with threshhold:", threshhold
    if X_test is not None:
        X_test.columns = X_all.columns
        X_all = pd.concat([X_test, X_all], axis=0, ignore_index=True)

    if X_valid is not None:
        X_valid.columns = X_all.columns
        X_all = pd.concat([X_valid, X_all], axis=0, ignore_index=True)

    c = X_all.corr().abs()

    corcols = {}
    for col in range(len(c.columns)):
        for row in range(len(c.index)):
            if c.columns[col] in corcols or c.index[row] in corcols:
                continue
            if row <= col:
                continue
            if c.iloc[row, col] < 1.0 and c.iloc[row, col] > threshhold:
                av1 = c.iloc[row, :].mean()  # row mean
                av2 = c.iloc[:, col].mean()  # col mean
                if av1 < av2:
                    key = c.columns[col]
                else:
                    key = c.index[row]
                corcols[key] = c.index[row] + " <> " + c.columns[col] + " :" + str(
                    "%4.3f mean: (%4.3f/%4.3f)" % (c.iloc[row, col], av1, av2))

    for el in corcols.keys():
        print "Dropped: %-32s due to: %32r" % (el, corcols[el])
    X_all = X_all.drop(corcols, axis=1)

    if X_test is not None:
        X_train = X_all[len(X_test.index):]
        X_test = X_all[:len(X_test.index)]
        return (X_train, X_test)

    if X_valid is not None:
        X_train = X_all[(len(X_valid) + len(X_test.index)):]
        X_test = X_all[len(X_valid):len(X_test.index)]
        X_valid = X_all[:len(X_valid)]
        return (X_train, X_test, X_valid)


    else:
        return X_all


def makePredictions(lmodel, lXs_test, lidx, filename):
    """
    Uses priorily fit model to make predictions
    """
    print "Saving predictions to: ", filename
    print "Final test dataframe:", lXs_test.shape
    preds = lmodel.predict_proba(lXs_test)[:, 1]
    pred_df = pd.DataFrame(preds, index=lidx, columns=['label'])
    pred_df.to_csv(filename)


def make_calibration(lmodel, Xtrain, ytrain):
    """
    Make Platt/isotonic calibration with sklearn
    """
    calli_clf = CalibratedClassifierCV(lmodel, method='isotonic', cv=3)
    calli_clf.fit(Xtrain, ytrain)
    print calli_clf.calibrated_classifiers_


def analyzeModel(lmodel, feature_names):
    """
    Analysis of data if feature_names are available
    """
    if hasattr(lmodel, 'coef_'):
        print("Analysis of data...")
        print("Dimensionality: %d" % lmodel.coef_.shape[1])
        print("Density: %4.3f" % density(lmodel.coef_))
        if feature_names is not None:
            top10 = np.argsort(lmodel.coef_)[0, -10:][::-1]
            # print model.coef_[top10b]
            for i in xrange(top10.shape[0]):
                print("Top %2d: coef: %0.3f %20s" % (i + 1, lmodel.coef_[0, top10[i]], feature_names[top10[i]]))


def sigmoid(z):
    """
    classical sigmoid
    """
    g = 1.0 / (1.0 + np.exp(-z));
    return (g)


# http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                 shape=loader['shape'])


def weightedGridsearch(lmodel, lX, ly, lw, fitWithWeights=False, n_folds=5, useProba=False, scale_wt='auto', n_jobs=1,
                       local_scorer='roc_auc'):
    """
    Uses sample weights and individual scoring function, used in Higgs challenge, needs modification cross_Validation.py
    """
    if not 'sample_weight' in inspect.getargspec(lmodel.fit).args:
        print("WARNING: Fit function ignores sample_weight!")

    fit_params = {}
    fit_params['scoring_weight'] = lw
    fit_params['fitWithWeights'] = fitWithWeights

    # parameters = {'n_estimators':[150,300], 'max_features':[5,10]}#rf
    # parameters = {'n_estimators':[250], 'max_features':[6,8,10],'min_samples_leaf':[5,10]}#xrf+xrf
    # parameters = {'max_depth':[7], 'learning_rate':[0.08],'n_estimators':[100,200,300],'subsample':[0.5],'max_features':[10],'min_samples_leaf':[50]}#gbm
    # parameters = {'max_depth':[7], 'learning_rate':[0.08],'n_estimators':[200],'subsample':[1.0],'max_features':[10],'min_samples_leaf':[20]}#gbm
    parameters = {'max_depth': [6], 'learning_rate': [0.1, 0.08, 0.05], 'n_estimators': [300, 500, 800],
                  'subsample': [1.0], 'loss': ['deviance'], 'min_samples_leaf': [100], 'max_features': [8]}  # gbm
    # parameters = {'max_depth':[10], 'learning_rate':[0.001],'n_estimators':[500],'subsample':[0.5],'loss':['deviance']}#gbm
    # parameters = {'max_depth':[15,20,25], 'learning_rate':[0.1,0.01],'n_estimators':[150,300],'subsample':[1.0,0.5]}#gbm
    # parameters = {'max_depth':[20,30], 'learning_rate':[0.1,0.05],'n_estimators':[300,500,1000],'subsample':[0.5],'loss':['exponential']}#gbm
    # parameters = {'max_depth':[15,20], 'learning_rate':[0.05,0.01,0.005],'n_estimators':[250,500],'subsample':[1.0,0.5]}#gbm
    # parameters = {'n_estimators':[100,200,400], 'learning_rate':[0.1,0.05]}#adaboost
    # parameters = {'filter__percentile':[20,15]}#naives bayes
    # parameters = {'filter__percentile': [15], 'model__alpha':[0.0001,0.001],'model__n_iter':[15,50,100],'model__penalty':['l1']}#SGD
    # parameters['model__n_neighbors']=[40,60]}#knn
    # parameters['model__alpha']=[1.0,0.8,0.5,0.1]#opt nb
    # parameters = {'n_neighbors':[10,30,40,50],'algorithm':['ball_tree'],'weights':['distance']}#knn
    clf_opt = grid_search.GridSearchCV(lmodel, parameters, n_jobs=n_jobs, verbose=1, scoring=local_scorer, cv=n_folds,
                                       fit_params=fit_params, refit=True)
    clf_opt.fit(lX, ly)
    # dir(clf_opt)
    for params, mean_score, scores in clf_opt.grid_scores_:
        print("%0.3f (+/- %0.3f) for %r" % (mean_score, scores.std(), params))

    scores = cross_validation.cross_val_score(lmodel, lX, ly, fit_params=fit_params, scoring=local_scorer, cv=n_folds)
    print "Score: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
    return (clf_opt.best_estimator_)


def buildModel(clf, lX, ly, cv=None, scoring=None, n_jobs=1, trainFull=False, verbose=False):
    if isinstance(lX, pd.DataFrame): lX = lX.values
    if isinstance(ly, pd.DataFrame) or isinstance(ly, pd.Series): ly = ly.values

    score = cross_validation.cross_val_score(clf, lX, ly, fit_params=None, scoring=scoring, cv=cv, n_jobs=n_jobs)
    if verbose:
        print "cv-score: %6.3f +/- %6.3f" % (score.mean(), score.std())
        print "all scores: %r" % (score)
    if trainFull:
        print "Train on all data..."
        clf.fit(lX, ly)
        return (clf)
    else:
        return score


def runningMean(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')


def buildXvalModel(clf_orig, lX_df, ly, sample_weight=None, class_names=None, refit=False, cv=None):
    """
  Final model building part
  """
    print "Training the model..."

    if isinstance(lX_df, pd.DataFrame):
        lX = lX_df.values
    else:
        lX = lX_df
    if isinstance(ly, pd.DataFrame) or isinstance(ly, pd.Series): ly = ly.values
    if isinstance(sample_weight, pd.Series): sample_weight = sample_weight.values

    ypred = np.zeros((len(ly),))
    score1 = np.zeros((len(cv), 1))
    score2 = np.zeros((len(cv), 1))
    for i, (train, test) in enumerate(cv):
        clf = clone(clf_orig)
        ytrain, ytest = ly[train], ly[test]

        sw = "nosw"
        if sample_weight is not None:
            sw = "sw"
            print sample_weight
            clf.fit(lX[train, :], ytrain, sample_weight=sample_weight[train])
        else:
            clf.fit(lX[train, :], ytrain)

        ypred[test] = clf.predict(lX[test, :])
        score1[i] = accuracy_score(ly[test], ypred[test])
        #score1[i] = mean_abs_percentage_error(np.expm1(ly[test]), np.expm1(ypred[test]))
        # rmse[i] = root_mean_squared_error(ly[test], ypred[test])
        # score2[i] = root_mean_squared_percentage_error_mod(ly[test], ypred[test])
        ypred[test] = clf.predict_proba(lX[test, :])[:,1]
        score2[i] = roc_auc_score(ly[test], ypred[test])

        print "train set: %2d samples: %5d/%5d rmse: %4.3f  mean: %4.3f %s " % (
        i, lX[train, :].shape[0], lX[test, :].shape[0], score2[i], score2[:i + 1].mean(), sw)
        # showMisclass(np.expm1(ly[test]),np.expm1(ypred[test]),lX[test,:],index=class_names.values[test],t=2.0)
    # if isinstance(lX_df,pd.DataFrame):
    #  showMisclass(ly,ypred,lX_df,index=class_names)
    # print classification_report(ly, ypred, target_names=class_names)

    print("ACC       :%6.3f +/-%6.3f" % (score1.mean(), score1.std()))
    print("AUC      :%6.3f +/-%6.3f" % (score2.mean(), score2.std()))

    # training on all data
    if refit:
        if sample_weight is not None:
            clf_orig.fit(lX, ly, sample_weight=sample_weight)
        else:
            clf_orig.fit(lX, ly)
    return (clf_orig)


def shuffleDF(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
        return df


def density(m):
    """
    For sparse & dense matrices
    """
    if isinstance(m, pd.DataFrame) or isinstance(m, np.ndarray):
        nz = np.count_nonzero(m.values)
        print "Non-zeros     : %12d" % (nz)
        te = m.shape[0] * m.shape[1]
        print "Total elements: %12d" % (te)
        print "Ratio         : %12.2f" % (float(nz) / float(te))
    else:
        entries = m.shape[0] * m.shape[1]
        print "Density      : %12.3f" % (m.nnz / float(entries))


def buildWeightedModel(lmodel, lXs, ly, lw=None, fitWithWeights=True, n_folds=8, useProba=True, scale_wt=None, n_jobs=1,
                       verbose=False, local_scorer='roc_auc'):
    """
    Build model using sample weights, can use weights for scoring function
    """

    fit_params = {}
    fit_params['scoring_weight'] = lw
    fit_params['fitWithWeights'] = fitWithWeights

    print "Xvalidation..."
    scores = cross_validation.cross_val_score(lmodel, lXs, ly, fit_params=fit_params, scoring=local_scorer, cv=n_folds,
                                              n_jobs=n_jobs)
    print "<SCORE>= %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
    print "Building model with all instances..."

    if fitWithWeights:
        print "Use sample weights for final model..."
        lmodel.fit(lXs, ly, sample_weight=lw)
    else:
        lmodel.fit(lXs, ly)

    # analysis of final predictions
    if useProba:
        print "Using predic_proba for final model..."
        probs = lmodel.predict_proba(lXs)[:, 1]
        # plot it
        plt.hist(probs, label='final model', bins=50, color='b')
        plt.legend()
        plt.draw()

    return (lmodel)


def compareList(uniq_train, uniq_test, verbose=True):
    """
    Comparing to lists
    """
    # uniq_train = Xtrain[col].unique()
    if verbose: print "Train - unique: %d %r:" % (uniq_train.shape[0], uniq_train)
    # uniq_test = Xtest[col].unique()
    if verbose: print "Test - unique: %d %r:" % (uniq_test.shape[0], uniq_test)
    isect = np.intersect1d(uniq_train, uniq_test)
    if verbose: print "Test - intersect: %d %r:" % (isect.shape[0], isect)
    only_train = np.in1d(uniq_train, isect, assume_unique=True, invert=True)
    if verbose: print "In Train only : %d %r:" % (uniq_train[only_train].shape[0], uniq_train[only_train])
    only_test = np.in1d(uniq_test, isect, assume_unique=True, invert=True)
    if verbose: print "In Test only : %d %r:" % (uniq_test[only_test].shape[0], uniq_test[only_test])
    return np.hstack((uniq_train[only_train], uniq_test[only_test]))


def group_binary_data(lX, degree=2, append=True):
    """
    This function groups binary data (i.e. after one hot encoding)
    Only for several categorical data!!! Better group several columns with categoricals!!!
    Alternatively one could group cols before and then one hot encode...
    """
    m, n = lX.shape
    new_data = None
    combs = itertools.combinations(range(n), degree)
    for indices in combs:
        print indices
        indices = lX.columns[list(indices)]
        print indices
        print lX[indices].head(30)
        if not isinstance(new_data, pd.DataFrame):
            new_data = pd.DataFrame(lX[indices].apply(np.multiply, axis=1))
        else:
            new_data = pd.concat([new_data, pd.DataFrame(lX[indices].apply(np.multiply, axis=1))], axis=1)

    if append:
        lX_reduced = pd.concat([lX,new_data],axis=1)
    else:
        lX_reduced = new_data
    print "New test data:", lX_reduced.shape
    return lX_reduced


def group_binary_sparse(lXs, lXs_test, degree=2, append=True):
    """
    multiply columns of sparse data
    """
    print "Columnwise min of data..."
    # also transform old data
    # (Xold,Xold_test) = linearFeatureSelection(model,Xold,Xold_test,5000)
    Xtmp = sparse.vstack((lXs_test, lXs), format="csc")
    Xtmp = pd.DataFrame(np.asarray(Xtmp.todense()))
    new_data = None
    m, n = Xtmp.shape
    for indices in itertools.combinations(range(n), degree):
        indices = Xtmp.columns[list(indices)]
        print indices
        if not isinstance(new_data, pd.DataFrame):
            new_data = pd.DataFrame(Xtmp[indices].apply(np.min, axis=1))
        else:
            new_data = pd.concat([new_data, pd.DataFrame(Xtmp[indices].apply(np.min, axis=1))], axis=1)
        print new_data.shape

    # making test data
    Xreduced_test = new_data[:lXs_test.shape[0]]
    if append:
        Xreduced_test = sparse.hstack((lXs_test, Xreduced_test), format="csr")
    print "New test data:", Xreduced_test.shape

    # making train data
    Xreduced = new_data[lXs.shape[0]:]
    if append:
        Xreduced = sparse.hstack((lXs, Xreduced), format="csr")
    print "New test data:", Xreduced.shape
    return (Xreduced, Xreduced_test)


def featureImportance(clf, X, y):
    """
    New function for GradientBoostingClassifier
    """
    print "New feature importance..."
    clf.fit(X, y)

    # Plot feature importance
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


def rfFeatureImportance(forest, Xold, Xold_test, n):
    """
    Selects n best features from a model which has the attribute feature_importances_ is it buggy?
    """
    print "Feature importance..."
    if not hasattr(forest, 'feature_importances_'):
        print "Missing attribute feature_importances_ ...leaving"
        # return
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)  # perhas we need it later

    indices = np.argsort(importances)[::-1]
    print indices
    # Print the feature ranking
    print("Feature ranking:")

    print("%3s %-30s %3s %10s %6s" % ("nr", "feature", "id", "importance", "std"))
    for i, f in enumerate(indices):
        print("%3d %-30s %3d %10.4f %6.4f" % (i + 1, Xold.columns[f], f, importances[f], std[f]))

    # Plot the feature importances of the forest
    plt.bar(left=np.arange(len(indices)), height=importances[indices], width=0.35, color='r', yerr=std[indices])
    plt.ylabel('Importance')
    plt.title("Feature importances")
    # stack train and test data
    Xreduced = pd.concat([Xold_test, Xold])
    # sorting features
    n = len(indices) - n
    print "Selection of ", n, " top features..."
    Xreduced = Xreduced.iloc[:, indices[0:n]]
    print Xreduced.columns
    # split train and test data
    # pd slicing sometimes confusing...last element in slicing is inclusive!!! use iloc for integer indexing (i.e. in case index are float or not ordered)
    pdrowidx = Xold_test.shape[0] - 1
    Xreduced_test = Xreduced[:len(Xold_test.index)]
    Xreduced = Xreduced[len(Xold_test.index):]
    # print "Xreduced_test:",Xreduced_test
    print "Xreduced_test:", Xreduced_test.shape
    print "Xreduced_train:", Xreduced.shape
    return (Xreduced, Xreduced_test)


def xgbFeatureImportance(clf, X, y):
    """
    Selects n best features, strange importances...
    """
    print "XGB Feature importance..."
    clf.fit(X, y)

    fmap = clf.get_fscore()
    importances = np.zeros(X.shape[1])

    for i, (k, v) in enumerate(fmap.iteritems()):
        idx = int(float(k.replace("f", "")))
        print "k", k
        print "feature", idx
        print "importance", v
        print "name:", X.columns[idx]
        # print indices[i]
        importances[idx] = v
    # print X.columns[indices[i]]
    indices = np.argsort(importances)[::-1]
    # indices = indices.astype(int)

    for i, f in enumerate(indices):
        print("%3d. feature %16s %3d - %6.4f" % (i, X.columns[f], f, importances[f]))

    print fmap
    fscore = [(v, k) for k, v in fmap.iteritems()]
    print "fscore", fscore.sort(reverse=True)

    # plt.bar(left=np.arange(len(indices)),height=importances[indices] , width=0.35, color='r')
    # plt.show()


def linearFeatureSelection(lmodel, Xold, Xold_test, n):
    """
    Analysis of data if coef_ are available for sparse matrices, better use t-scores
    """
    print "Selecting features based on important coefficients..."
    if hasattr(lmodel, 'coef_') and isinstance(Xold, sparse.csr.csr_matrix):
        print("Dimensionality before: %d" % lmodel.coef_.shape[1])
        indices = np.argsort(lmodel.coef_)[0, -n:][::-1]
        # print model.coef_[top10b]
        # for i in xrange(indices.shape[0]):
        #    print("Top %2d: coef: %0.3f col: %2d" % (i+1,lmodel.coef_[0,indices[i]], indices[i]))
        plt.bar(left=np.arange(len(indices)), height=lmodel.coef_[0, indices], width=0.35, color='r')
        plt.ylabel('Importance')
        # stack train and test data
        # Xreduced=np.vstack((Xold_test,Xold))
        Xreduced = sparse.vstack((Xold_test, Xold), format="csr")
        # sorting features
        # print indices[0:n]
        Xtmp = Xreduced[:, indices[0:n]]
        print("Dimensionality after: %d" % Xtmp.shape[1])
        # split train and test data
        Xreduced_test = Xtmp[:Xold_test.shape[0]]
        Xreduced = Xtmp[Xold_test.shape[0]:]
        return (Xreduced, Xreduced_test)


# The step callback function, this function
# will be called every step (generation) of the GA evolution
def evolve_callback(ga_engine):
    generation = ga_engine.getCurrentGeneration()
    if generation % 1 == 0:
        pop = ga_engine.getPopulation()
        column_names = pop.oneSelfGenome.getParam("Xtrain").columns
        binary_list = [True if value == 1 else False for value in ga_engine.bestIndividual()]
        print "Best individual:", column_names[binary_list]
        print "Best individual:", ga_engine.bestIndividual()
    return False


def eval_func(genome):
    """
  Evaluation function for GA
  """
    model = genome.getParam("model")
    Xtrain = genome.getParam("Xtrain")
    ytrain = genome.getParam("ytrain")
    cv = genome.getParam("cv")
    scoring_func = genome.getParam("scoring_func")
    n_jobs = genome.getParam("n_jobs")

    # print genome
    binary_list = [True if value == 1 else False for value in genome]
    Xact = Xtrain.iloc[:, binary_list]
    # print "New shape:",Xact.shape
    # print "Columns:",Xact.columns
    t0 = time()
    score = buildModel(model, Xact, ytrain, cv=cv, scoring=scoring_func, n_jobs=n_jobs, trainFull=False)
    run_time = time() - t0

    # print "cv-score: %6.3f +/- %6.3f genome: %s" %(-1*score.mean(),score.std(),[value for value in genome])
    print "cv-score: %6.3f +/- %6.3f cv-runs: %4d time: %4.2f" % (-1 * score.mean(), score.std(), len(score), run_time)
    return -1 / score.mean()


def genetic_feature_selection(model, Xtrain, ytrain, Xtest, pool_features=None, start_features=None,
                              scoring_func='mean_squared_error', cv=None, n_iter=3, n_pop=20, n_jobs=1):
    """
    Genetic feature selection
    """
    from pyevolve import G1DBinaryString
    from pyevolve import Initializators, Mutators
    from pyevolve import GSimpleGA
    from pyevolve import Selectors
    from pyevolve import Statistics
    from pyevolve import DBAdapters
    from pyevolve.GenomeBase import GenomeBase

    print cv
    print model

    if pool_features == None:
        pool_features = Xtrain.columns

    genome = G1DBinaryString.G1DBinaryString(Xtrain.shape[1])

    genome.evaluator.set(eval_func)
    # genome.mutator.set(Mutators.G1DBinaryStringMutatorFlip)

    # start_features='11111111111111111111111111101111101111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111101111111111111111111111111111110111111111111111111111111111111111101111111111111111110111111111111111111011111111111111111011110111111111111111101111011111111111111011111111111111111111111111011111111111100111111111111101111111101111111111111011111111111111111111111111111111111111111011111111111101111111111111111111111111111111111111111111111111101111111111101111111111101111111111111111011111111111111111111111111111111111111111111111111111011111111111111111111111111110111111110111111111111111111111111111111111111111111111111111111111111111011111111111111111111111101111111111011110111111111011110111001111101101101001100111011110010000101000110011111111010000001110110100111101000000001111111110010010000111100100010100011011001001011100101111010000001110001001011110011101101100001111101110010001000010100111001010011011110101100100011111101001111001000010000100100111111111100111110000101001000000101001100111011110010000100101001110010101100010100101101101001000110100111001000010111101111000110001110110101101111010110010101010011010100110100111110001110101011100010011100100000011000100101010111110001010001010010111000000110101011000100110000000100110000000010101101111001001100111011011110110101111111011011100111110010110100101110'
    # if isinstance(start_features,str):
    # binary_list = [True if value=='1' else False for value in start_features]
    # start_features = list(Xtrain.columns[binary_list])

    genome.setParams(model=model, Xtrain=Xtrain, ytrain=ytrain, start_features=start_features,
                     pool_features=pool_features, scoring_func=scoring_func, cv=cv, n_jobs=n_jobs)
    genome.initializator.set(Initializators.G1DBinaryStringInitializator)
    # Genetic Algorithm Instance
    ga = GSimpleGA.GSimpleGA(genome)
    ga.stepCallback.set(evolve_callback)
    # GSimpleGA.GSimpleGA.setMultiProcessing(ga)
    ga.selector.set(Selectors.GTournamentSelector)
    ga.setGenerations(n_iter)
    ga.setPopulationSize(n_pop)
    # Do the evolution, with stats dump
    # frequency of 10 generations
    ga.evolve(freq_stats=1)

    # Best individual
    # print ga.bestIndividual()
    binary_list = [True if value == 1 else False for value in ga.bestIndividual()]
    print "Best features:"
    print Xtrain.columns[binary_list]
    print "Selected %d out of %d features" % (sum(binary_list), Xtrain.shape[1])
    buildModel(model, Xtrain.iloc[:, binary_list], ytrain, cv=cv, scoring=scoring_func, n_jobs=n_jobs, trainFull=False,
               verbose=True)


def greedyFeatureSelection(lmodel, lX, ly, itermax=10, itermin=5, pool_features=None, start_features=None,
                           verbose=False, cv=5, n_jobs=4, scoring_func='mean_squared_error'):
    features = []

    if pool_features is None:
        pool_features = lX.columns

    if start_features is not None:
        features = start_features
    else:
        features = [col for col in lX.columns if not col in pool_features]

    scores = []
    score_opt = 1E10
    for i in xrange(itermax):
        print "Round %4d" % (i)
        score_best = 1E9
        for a, k in enumerate(xrange(len(pool_features))):
            act_feature = pool_features[k]
            if act_feature in set(features): continue

            features.append(act_feature)

            t0 = time()
            score = cross_validation.cross_val_score(lmodel, lX.loc[:, features], ly, fit_params=None,
                                                     scoring=scoring_func, cv=cv, n_jobs=n_jobs)
            run_time = time() - t0

            if 'mean_squared_error' in str(scoring_func):
                score = -1 * (score) ** 0.5
            else:
                score = -1 * score

            if verbose:
                print "(%4d/%4d) TARGET: %-12s - <score>= %0.4f (+/- %0.5f) score,iteration best= %0.4f score,overall best: %0.4f features: %5d time: %6.2f" % (
                    a + 1, len(pool_features), act_feature, score.mean(), score.std(), score_best, score_opt,
                    lX.loc[:, features].shape[1], run_time)

            if score.mean() < score_best:
                score_best = score.mean()
                new_feat = act_feature
                features_best = features[:]
            del features[-1]
        features.append(new_feat)
        scores.append(score_best)

        if (i > itermin and (score_opt > score_best)):
            print "Converged with threshold: %0.6f" % (np.abs(score_opt - score_best))
            score_opt = score_best
            opt_list = features_best
            break
        if score_best < score_opt:
            score_opt = score_best
            opt_list = features_best

        print " nr features: %5d " % (len(features)),
        print " score,iteration best= %0.4f score,overall best: %0.4f \n%r" % (score_best, score_opt, features)

    print "Scores:", scores

    print "Best score: %6.4f with %5d features:\n%r" % (score_opt, len(opt_list), opt_list)
    plt.plot(scores)
    plt.show()


def iterativeFeatureSelection(lmodel, Xold, Xold_test, ly, iterations=5, nrfeats=1, scoring=None, cv=None, n_jobs=8):
    """
    Iterative feature selection e.g. via random Forest
    """
    print "Iterative features selection"
    for i in xrange(iterations):
        print ">>>Iteration: ", i, "<<<"
        lmodel = buildModel(lmodel, Xold, ly, cv=cv, scoring=scoring, n_jobs=n_jobs, trainFull=True, verbose=True)
        # lmodel.fit(Xold,ly)
        (Xold, Xold_test) = rfFeatureImportance(lmodel, Xold, Xold_test, nrfeats)
        # Xold.to_csv("../stumbled_upon/data/Xlarge_"+str(i)+".csv")
        # Xold_test.to_csv("../stumbled_upon/data/XXlarge_test_"+str(i)+".csv")
    return (Xold, Xold_test)


def removeInstances(lXs, ly, preds, t, returnSD=True):
    """
    Removes examples from train set either due to prediction error or due to standard deviation
    Preds should come from repeated CV.
    """
    if returnSD:
        res = preds
    else:
        res = np.abs(ly - preds)
    d = {'abs_err': pd.Series(res)}
    res = pd.DataFrame(d)
    res.index = lXs.index
    lXs_reduced = pd.concat([lXs, res], axis=1)
    boolindex = lXs_reduced['abs_err'] < t
    lXs_reduced = lXs_reduced[boolindex]
    # print lXs_reduced.shape
    # ninst[i]=len(Xtrain.index)-len(lXs_reduced.index)
    lXs_reduced = lXs_reduced.drop(['abs_err'], axis=1)
    # print "New dim:",lXs_reduced.shape
    ly_reduced = ly[np.asarray(boolindex)]
    return (lXs_reduced, ly_reduced)

def removeLowFreq(df, column_names, removeRare_freq = 2, discard_rows = False, fillNA = 9999):
    """
    Remove categories with frequency lowe than threshold
    """
    print "Remove rare features based on frequency..."
    for col in column_names:
        ser = df[col]
        counts = ser.value_counts().keys()
        idx = ser.value_counts() > removeRare_freq
        threshold = idx.astype(int).sum()
        print "%s has %d different values before, min freq: %d - threshold %d" % (
            col, len(counts), removeRare_freq, threshold)
        if len(counts) > threshold:
            if discard_rows:
                print "Removing rows!"
                print df.shape
                df = df[ser.isin(counts[:threshold])]
                print df.shape
            else:
                ser[~ser.isin(counts[:threshold])] = fillNA
        if len(counts) <= 1:
            print("Dropping Column %s with %d values" % (col, len(counts)))
            df = df.drop(col, axis=1)
        #else:
        #    df[col] = ser.astype('category')
        counts = df[col].value_counts().keys()
        print "%s has %d different values after" % (col, len(counts))

    return df

def removeLowVar(X_all, threshhold=1E-5):
    """
    remove useless data
    """
    print X_all.std()

    if isinstance(X_all, sparse.csc_matrix):
        print "Making matrix dense again..."
        X_all = pd.DataFrame(X_all.toarray())

    idx = np.asarray(X_all.std() <= threshhold)
    # for col in X_all.columns[idx]:
    # print "Column:",col
    # print X_all[[col]].describe()
    # raw_input()

    if len(X_all.columns[idx]) > 0:
        print "Dropped %4d zero variance columns (threshold=%6.3f): %r" % (
            np.sum(idx), threshhold, list(X_all.columns[idx]).sort())
        X_all.drop(X_all.columns[idx], axis=1, inplace=True)
    else:
        print "Variance filter dropped nothing (threshhold = %6.3f)." % (threshhold)

    return (X_all)


def pcAnalysis(X, Xtest, y=None, w=None, ncomp=2, transform=False, classification=False):
    """
    PCA
    """
    pca = PCA(n_components=ncomp)
    if transform:
        print "PC reduction"
        X_all = pd.concat([Xtest, X])

        X_r = pca.fit_transform(np.asarray(X_all))
        print(pca.explained_variance_ratio_)
        # split
        X_r_train = X_r[len(Xtest.index):]
        X_r_test = X_r[:len(Xtest.index)]
        return (X_r_train, X_r_test)

    elif classification:
        print "PC analysis for classification"
        X_all = pd.concat([Xtest, X])
        # this is transformation is necessary otherwise PCA gives rubbish!!
        ytrain = np.asarray(y)

        X_r = pca.fit_transform(np.asarray(X_all))

        if w is None:
            plt.scatter(X_r[ytrain == 0, 0], X_r[ytrain == 0, 1], c='r', label="1", alpha=0.1)
            plt.scatter(X_r[ytrain == 1, 0], X_r[ytrain == 1, 1], c='g', label="0", alpha=0.1)
        else:
            plt.scatter(X_r[ytrain == 0, 0], X_r[ytrain == 0, 1], c='r', label="background", s=w[ytrain == 0] * 25.0,
                        alpha=0.1)
            plt.scatter(X_r[ytrain == 1, 0], X_r[ytrain == 1, 1], c='g', label="signal", s=w[ytrain == 1] * 1000.0,
                        alpha=0.1)

        print(pca.explained_variance_ratio_)
        plt.legend()
        # plt.xlim(-3500,2000)
        # plt.ylim(-1000,2000)
        plt.draw()
    else:
        print "PC analysis for train/test"
        X_all = pd.concat([Xtest, X])
        X_r = pca.fit_transform(np.asarray(X_all))
        plt.scatter(X_r[len(Xtest.index):, 0], X_r[len(Xtest.index):, 1], c='r', label="train", alpha=0.5)
        plt.scatter(X_r[:len(Xtest.index), 0], X_r[:len(Xtest.index), 1], c='g', label="test", alpha=0.5)
        print("Explained variance:", pca.explained_variance_ratio_)
        plt.legend()
        plt.show()


def root_mean_squared_percentage_error(ytrue, y, factor=1.0):
    assert (len(ytrue) == len(y))
    if factor > -1.0:
        # print "Scaling y with %4.3f"%(factor)
        y = y * factor

    if y.shape != ytrue.shape:
        ytrue = ytrue.flatten()
        y = y.flatten()

    err = np.power((y - ytrue) / ytrue, 2)
    err[np.where(ytrue < 1E-15)] = 0.0
    err = np.sqrt(np.mean(err))
    return err


def mean_abs_percentage_error(ytrue, ypred):
    return np.mean(np.abs((ypred / ytrue - 1)))


def root_mean_squared_percentage_error_old(ytrue, ypred):
    return np.sqrt(np.mean((ypred / ytrue - 1) ** 2))


def root_mean_squared_percentage_error_mod(ytrue, ypred):
    ytrue = np.expm1(ytrue)
    ypred = np.expm1(ypred)
    return root_mean_squared_percentage_error(ytrue, ypred, factor=0.985)


def root_mean_squared_log_error(x, y):
    x = np.clip(x, a_min=0.0, a_max=1E15)
    y = np.clip(y, a_min=0.0, a_max=1E15)
    if ((y + 1.0) < 0.0).sum() > 0 or ((x + 1.0) < 0.0).sum():
        print "WARNING!"
        print "x", ((x + 1.0) < 0.0).sum()
        print "y", ((y + 1.0) < 0.0).sum()
        idx = (y + 1.0) < 0.0
        print y[idx]

    x = np.log(x + 1.0)
    y = np.log(y + 1.0)

    return root_mean_squared_error(x, y)


def root_mean_squared_error(x, y):
    mse = mean_squared_error(x, y)
    return mse ** 0.5


def mean_absolute_error(x, y):
    x = x.flatten()
    y = y.flatten()
    return np.mean(np.abs(x - y))


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    return log_loss(y_true, y_pred, eps=eps, normalize=True)


def getOOBCVPredictions(lmodel, lXs, ly, repeats=1, cv=5, returnSD=True, score_func='rmse'):
    """
    Get cv oob predictions for classifiers
    """
    funcdict = {}
    if score_func == 'rmse':
        funcdict['scorer_funct'] = root_mean_squared_error
    else:
        funcdict['scorer_funct'] = roc_auc_score

    print "Computing oob predictions..."
    oobpreds = np.zeros((lXs.shape[0], repeats))
    for j in xrange(repeats):
        # print lmodel.get_params()
        # cv = KFold(lXs.shape[0], n_folds=folds,random_state=j,shuffle=True)
        scores = np.zeros(len(cv))
        for i, (train, test) in enumerate(cv):
            Xtrain = lXs.iloc[train]
            Xtest = lXs.iloc[test]
            # print Xtest['avglinksize'].head(3)
            lmodel.fit(Xtrain, ly[train])
            if score_func == 'rmse':
                oobpreds[test, j] = lmodel.predict(Xtest)
                scores[i] = funcdict['scorer_funct'](ly[test], oobpreds[test, j])
            else:
                oobpreds[test, j] = lmodel.predict_proba(Xtest)[:, 1]
                scores[i] = funcdict['scorer_funct'](ly[test], oobpreds[test, j])
                # print "AUC: %0.2f " % (scores[i])
                # save oobpredictions
        print "Iteration:", j,
        print " <score>: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()),

        oobscore = funcdict['scorer_funct'](ly, oobpreds[:, j])
        print " score,oob: %0.3f" % (oobscore)

    scores = [funcdict['scorer_funct'](ly, oobpreds[:, j]) for j in xrange(repeats)]
    # simple averaging of blending
    oob_avg = np.mean(oobpreds, axis=1)
    print "Summary: <score,oob>: %0.3f (%d repeats)" % (funcdict['scorer_funct'](ly, oob_avg), repeats,)
    if returnSD:
        oob_avg = np.std(oobpreds, axis=1)
    return (oob_avg)


def lofFilter(pred, threshhold=10.0, training=True):
    """
    filter data according to local outlier frequency as computed by R...did not work
    """
    indices = []
    global test_indices
    lof = pd.read_csv("../stumbled_upon/data/lof.csv", sep=",", index_col=0)
    lof = lof[len(test_indices):]
    avg = np.mean(pred)
    for i in xrange(len(lof.index)):
        # print lof.iloc[i,0]
        if lof.iloc[i, 0] > threshhold:
            pred[i] = avg
            indices.append(i)
    # print indices
    print "threshhold:,", threshhold, "n,changed:", len(indices), " mean:", avg
    return pred


def filterClassNoise(lmodel, lXs, lXs_test, ly):
    """
    Removes training samples which could be class noise
    Done in outer XVal loop
    precision: Wieviel falsche habe ich erwischt
    recall: wieviele richtige sind durch die Lappen gegangen
    """
    threshhold = [0.045, 0.04, 0.035]
    folds = 10
    print "Filter strongly misclassified classes..."
    # rdidx=random.sample(xrange(1000), 20)
    # print rdidx
    # lXs = lXs.iloc[rdidx]
    # ly = ly[rdidx]
    preds = getOOBCVPredictions(lmodel, lXs, lXs_test, ly, folds, 10)
    print preds
    plt.hist(preds, bins=20)
    plt.show()
    # print "stdev:",std
    # should be oob or cvalidated!!!!
    # preds = lmodel.predict_proba(lXs)[:,1]
    scores = np.zeros((folds, len(threshhold)))
    oobpreds = np.zeros((lXs.shape[0], folds))
    for j, t in enumerate(threshhold):
        # XValidation
        cv = KFold(lXs.shape[0], n_folds=folds, indices=True, random_state=j, shuffle=True)
        ninst = np.zeros(folds)
        for i, (train, test) in enumerate(cv):
            Xtrain = lXs.iloc[train]
            ytrain = ly[train]
            # now remove examples from train
            lXs_reduced, ly_reduced = removeInstances(Xtrain, ytrain, preds[train], t)
            ninst[i] = len(Xtrain.index) - len(lXs_reduced.index)
            lmodel.fit(lXs_reduced, ly_reduced)

            # testing data, not manipulated
            Xtest = lXs.iloc[test]
            oobpreds[test, j] = lmodel.predict_proba(Xtest)[:, 1]

            scores[i, j] = roc_auc_score(ly[test], oobpreds[test, j])

        print "Threshhold: %0.3f  <AUC>: %0.3f (+/- %0.3f) removed instances: %4.2f" % (
            t, scores[:, j].mean(), scores[:, j].std(), ninst.mean()),
        print " AUC oob: %0.3f" % (roc_auc_score(ly, oobpreds[:, j]))
    scores = np.mean(scores, axis=0)
    print scores
    plt.plot(threshhold, scores, 'ro')
    top = np.argsort(scores)
    optt = threshhold[top[-1]]
    print "Optimum threshhold %4.2f index: %d with score: %4.4f" % (optt, top[-1], scores[top[-1]])
    lXs_reduced, ly_reduced = removeInstances(lXs, ly, preds, optt)
    return (lXs_reduced, lXs_test, ly_reduced)


def showMisclass(ly, preds, lXs, index=None, model=None, t=1.0, bubblesizes=None):
    """
    Show bubble plot of strongest misclassifications...
    """
    preds = preds.flatten()

    print "Show strongly misclassified classes..."
    if model is not None:
        folds = 4
        repeats = 1
        preds = getOOBCVPredictions(model, lXs, ly, folds, repeats, returnSD=False)

    abs_err = pd.DataFrame({'abs_err': pd.Series(np.abs(ly - preds))})
    perc_err = pd.DataFrame({'abs_perc_err': pd.Series(np.abs(ly - preds) / ly)})
    sq_percerr = pd.DataFrame({'sq_percerr': pd.Series((ly - preds) / ly) ** 2})
    residue = pd.DataFrame({'residue': pd.Series((ly - preds))})
    ly = pd.DataFrame({'y': pd.Series(ly)})
    preds = pd.DataFrame({'preds': pd.Series(preds)})

    lXs_plot = pd.concat([ly, preds, residue, abs_err, perc_err, sq_percerr], axis=1)
    if index is None:
        lXs_plot.index = lXs.index
    else:
        lXs_plot.index = index
    # lXs_plot=pd.concat([lXs_plot,lXs], axis=1)
    lXs_plot.sort(columns='sq_percerr', inplace=True, ascending=False)
    # lXs_plot.sort_index(inplace=True)

    boolindex = lXs_plot['abs_perc_err'] > t

    lXs_plot = lXs_plot[boolindex]
    print "Number of instances left: %6d with threshold %f" % (lXs_plot.shape[0], t)
    col1 = 'preds'
    # col2 = 'residue'
    col2 = 'y'
    # bubblesizes=lXs_plot['y']*50
    bubblesizes = 30

    sct = plt.scatter(lXs_plot[col1], lXs_plot[col2], c=lXs_plot['abs_err'], s=bubblesizes, linewidths=2,
                      edgecolor='black')
    # sct = plt.scatter(lXs_plot[col1], lXs_plot[col2],s=bubblesizes, linewidths=2, edgecolor='black')
    sct.set_alpha(0.75)

    print "%4s %10s %8s %8s %8s %8s %8s %8s" % (
    "nr", "index", 'y', 'preds', 'residue', 'abs_error', '%%errorabs', 'sq_percerr')
    for i, (row_index, row) in enumerate(lXs_plot.iterrows()):
        plt.text(row[col1], row[col2], row_index, size=10, horizontalalignment='center')
        print "%4d %10s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f" % (
        i, row_index, row['y'], row['preds'], row['residue'], row['abs_err'], row['abs_perc_err'], row['sq_percerr'])
        if i > 100: break
    print "%4s %10s %6s %6s %8s %8s %8s %8s" % (
    "nr", "index", 'y', 'preds', 'residue', 'abs_error', '%%errorabs', 'sq_percerr')

    print "MEAN:\n", lXs_plot.describe()
    print "MEAN:\n", lXs_plot.iloc[1:, :].describe()

    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("error")
    plt.draw()
    lXs_plot.abs_perc_err.hist(bins=50)

    plt.show()


def scaleData(lXs, lXs_test=None, cols=None, normalize=False, epsilon=0.0):
    """
    standard scaling of data, also possible with sklearn StandardScaler but not with dataframe
    """
    if cols is None:
        cols = lXs.columns

    if lXs_test is not None:
        lX_all = pd.concat([lXs_test, lXs])
    else:
        lX_all = lXs

    if normalize:
        print "Normalize data..."
        lX_all[cols] = (lX_all[cols] - lX_all[cols].min()) / (lX_all[cols].max() - lX_all[cols].min())
    # standardize
    else:
        if epsilon > 1E-15:
            print "Standardize data with epsilon:", epsilon
            lX_all[cols] = (lX_all[cols] - lX_all[cols].mean()) / np.sqrt(lX_all[cols].var() + epsilon)
        else:
            print "Standardize data"
            lX_all[cols] = (lX_all[cols] - lX_all[cols].mean()) / lX_all[cols].std()

    # print lX_all[cols].describe()

    if lXs_test is not None:
        lXs = lX_all[len(lXs_test.index):]
        lXs_test = lX_all[:len(lXs_test.index)]
        return (lXs, lXs_test)
    else:
        return lX_all


def data_binning(X, binning):
    """
    Bin dat in n=binning bins
    """
    for col in X.columns:
        # print Xall[col]
        tmp = pd.cut(X[col].values, binning, labels=False)
        X[col] = np.asarray(tmp)
        # print Xall[col]
        # raw_input()
        # groups = Xall.groupby(tmp)
        # print groups
        # print groups.describe()

    return X


def binarizeProbs(a, cutoff):
    """
    turn probabilities to 1 and 0
    """
    if a > cutoff:
        return 1.0
    else:
        return 0.0


def make_polynomials(Xtrain, Xtest=None, degree=2, cutoff=100,quadratic=True):
    """
    Generate polynomial features
    """
    if Xtest is not None: Xtrain = Xtrain[len(Xtest.index):]
    m, n = Xtrain.shape
    if quadratic:
        indices = list(itertools.combinations_with_replacement(range(n), degree))
    else:
        indices = list(itertools.combinations(range(n), degree))
    new_data = []
    colnames = []
    for i, j in indices:
        stdout.write("\r%d out of %d" %(i, n))
        stdout.flush()
        name = str(Xtrain.columns[i]) + "_x_" + str(Xtrain.columns[j])
        Xnew = (Xtrain.values[:, i] * Xtrain.values[:, j])
        n_nonnull = (Xnew != 0).astype(int).sum()
        if n_nonnull > cutoff:
            new_data.append(Xnew)
            colnames.append(name)
        else:
            pass
            #print "Dropped:", name

    stdout.write("\n")
    new_data = pd.DataFrame(np.array(new_data).T, columns=colnames)
    return new_data


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def makeGridSearch(lmodel, lX, ly, n_jobs=1, refit=True, cv=5, scoring='roc_auc', random_iter=-1, parameters={}):
    print "Start GridSearch..."
    if parameters is None:
        # parameters = {'C':[10],'gamma':np.logspace(-5.0,-1.0,num=5)}
        parameters = {'C': [5, 10, 20], 'gamma': [0.004]}
        # parameters = {'filter__param': [30,50],'model__C':[1]}
        # parameters = {'pca__n_components':[30],'model__C':[1]}
        # parameters = {'model__C':[1,0.1,0.01,0.001]}#Linear SVC+LOGREG
        # parameters = {'C':[10]}#Linear SVC+LOGREG
        # parameters = {'C':[1.0],'penalty':['l2']}#Linear SVC+LOGREG
        # parameters = {'n_estimators':[50],'alpha_L1':[1E-1],'lambda_L2':[1E-1]}#XGBOOST GBLINEAR
        # parameters = {'alpha':[1E-6,1E-8],'n_iter':[100],'penalty':['l2']}#SGD
        # parameters = {'alpha':[1,1E-2,1E-4],'n_iter':[250],'penalty':['l2']}#SGD
        # parameters = {'hidden1_num_units': [600],'dropout1_p':[0.0,0.1,0.2,0.3,0.4,0.5],'maxout1_ds':[2,3,4],'hidden2_num_units': [600],'dropout2_p':[0.0],'maxout2_ds':[2,3,4],'hidden3_num_units': [600],'dropout3_p':[0.0],'maxout3_ds':[2,3,4],'max_epochs':[50,100,150],'update_learning_rate':[0.001,0.002,0.004]}#Lasagne
        # parameters = {'hidden1_num_units': [300],'dropout1_p':[0.25,0.5],'hidden2_num_units': [300],'dropout2_p':[0.5,0.25],'update_learning_rate':[0.01,0.008],'objective_alpha':[1E-10]}#Lasagne
        # parameters = {'hidden1_num_units': [500],'dropout1_p':[0.5],'hidden2_num_units': [500],'dropout2_p':[0.5,0.0],'hidden3_num_units': [500],'dropout3_p':[0.5,0.0],'max_epochs':[150,100],'objective_alpha':[1E-3,1E-6,1E-9],'update_learning_rate':[0.004,0.003,0.002]}#Lasagne
        # parameters = {'hidden1_num_units': [500,1000],'update_learning_rate':[0.0001,0.0005],'max_epochs':[500],'dropout1_p':[0.0,.2]}#Lasagne
        # parameters = {'hidden1_num_units': [500,500],'update_learning_rate':[0.0001,0.0005],'max_epochs':[500],'dropout1_p':[0.0,.2]}#Lasagne
        # parameters = {'hidden1_num_units': [200,300],'max_epochs':[1000],'dropout1_p':[0.1,0.2,0.3]}#Lasagne
        # parameters = {'n_estimators':[600], 'max_features':[16,20,24],'max_depth':[None],'max_leaf_nodes':[None],'min_samples_leaf':[1],'min_samples_split':[2,10],'criterion':['gini']}#xrf+xrf
        # parameters = {'class_weight': [{0: 1.,1: 1., 2: 1.,3: 1.,4: 1.,5: 1.,6: 1.,7: 1.,8: 1.,9: 1.},{0: 2.,1: 1., 2: 2.,3: 1.,4: 1.,5: 1.,6: 1.,7: 1.,8: 1.,9: 1.}]}
        # parameters = {'n_estimators':[300,400],'max_depth':[8,9,10],'learning_rate':[0.015,0.02,0.025,0.03],'subsample':[0.5,1.0]}#XGB+GBC
        # parameters = {'n_estimators':[400],'max_depth':[10],'learning_rate':[0.1,0.05,0.01],'subsample':[0.5]}#XGB+GBC

    if random_iter < 0:
        search = grid_search.GridSearchCV(lmodel, parameters, n_jobs=n_jobs, verbose=2, scoring=scoring, cv=cv,
                                          refit=refit)
    else:
        search = grid_search.RandomizedSearchCV(lmodel, param_distributions=parameters, n_jobs=n_jobs, verbose=2,
                                                scoring=scoring, cv=cv, refit=refit, n_iter=random_iter)

    search.fit(lX, ly)
    best_score = 1.0E5
    print("%6s %6s %6s %r" % ("OOB", "MEAN", "SDEV", "PARAMS"))
    for params, mean_score, cvscores in search.grid_scores_:
        oob_score = mean_score
        cvscores = cvscores
        mean_score = cvscores.mean()
        print("%6.3f %6.3f %6.3f %r" % (oob_score, mean_score, cvscores.std(), params))
    # if mean_score < best_score:
    #    best_score = mean_score
    #    scores[i,:] = cvscores

    # report(search.grid_scores_)

    if refit:
        return search.best_estimator_
    else:
        return None


def df_info(X):
    if isinstance(X, pd.DataFrame): X = X.values
    print "Shape:", X.shape, " size (MB):", float(X.nbytes) / 1.0E6, " dtype:", X.dtype


def analyzeLearningCurve(model, X, y, cv=8, score_func='roc_auc'):
    """
    make a learning curve according to http://scikit-learn.org/dev/auto_examples/plot_learning_curve.html
    """
    plot_learning_curve(model, "learning curve", X, y, ylim=(0.1, 1.01), cv=cv, n_jobs=1, scoring=score_func)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, scoring=f1_score,
                        train_sizes=np.linspace(.01, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring,
                                                            train_sizes=train_sizes)

    print "train_scores:",train_scores
    print "test_scores:",test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating = None
    max_rating = None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)

    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


def differentiateFeatures(X,plotting=False):
    """
    make derivative
    """
    print "Making 1st derivative..."
    tutto=[]

    for i, ind in enumerate(X.index):
        stdout.write("\r%d out of %d" %(i, len(list(X.index))))
        stdout.flush()
        row=[]
        row.append(ind)
        for el in np.gradient(X.ix[ind,:].values):
            row.append(el)
        tutto.append(row)
    newdf = pd.DataFrame(tutto).set_index(0)
    colnames = [ "diff"+str(x) for x in xrange(newdf.shape[1]) ]
    newdf.columns=colnames
    if plotting:
        newdf.iloc[3,:].plot()
        plt.show()

    stdout.write("\n")
    #print newdf.head(10)
    ###print newdf.describe()
    return(newdf)


def LeavePLabelOutWrapper(str_labels, n_folds=8, p=1, verbose=True):
    lenc = preprocessing.LabelEncoder()
    labels = lenc.fit_transform(str_labels) % n_folds
    cv = LeavePLabelOut(labels, p=1)
    if verbose: print "Labels: %r length %d" % (np.unique(labels).shape, len(cv))
    return cv


class KLabelFolds():
    def __init__(self, labels, n_folds=3, repeats=5):

        self.n_folds = n_folds
        self.repeats = repeats

        lenc = preprocessing.LabelEncoder()
        self.labels = lenc.fit_transform(labels)

    def __len__(self):
        return self.n_folds * self.repeats

    def __iter__(self):

        unique_labels = np.unique(self.labels)
        for i in range(self.repeats):
            cv = cross_validation.KFold(len(unique_labels), self.n_folds)
            unique_labels = shuffle(unique_labels)  # reproducible?
            for train, test in cv:
                test_labels = unique_labels[test]
                test_mask = np.in1d(self.labels, test_labels)
                train_mask = np.logical_not(test_mask)
                tr = np.where(train_mask)[0]
                # print tr
                # tr = shuffle(tr)
                # print tr
                yield (tr, np.where(test_mask)[0])


class ScaleContinuousOnly(BaseEstimator, TransformerMixin):
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        self.columns = X.columns
        self.binary_features = X.columns[(X.dtypes == np.int64) & (X.max() == 1) & (X.min() == 0)]
        self.continuous_features = X[[c for c in X.columns if c not in self.binary_features]].columns
        self.standard_scaler = StandardScaler()
        self.standard_scaler.fit(X[self.continuous_features])
        return self

    def transform(self, X, y=None, copy=None):
        scaled_df = pd.DataFrame(self.standard_scaler.transform(X[self.continuous_features]), index=X.index,
                                 columns=self.continuous_features)
        scaled_df = pd.concat([scaled_df, X[self.binary_features]], axis=1)[self.columns]
        return scaled_df


# some global vars
funcdict = {}
funcdict['rmse'] = root_mean_squared_error
funcdict['rmsle'] = root_mean_squared_log_error
funcdict['rmspe'] = root_mean_squared_percentage_error
funcdict['rmspe_exp1m'] = root_mean_squared_percentage_error_mod
funcdict['auc'] = roc_auc_score
funcdict['mae'] = mean_absolute_error
funcdict['msq'] = mean_squared_error
funcdict['log_loss'] = multiclass_log_loss
funcdict['accuracy_score'] = accuracy_score
funcdict['quadratic_weighted_kappa'] = quadratic_weighted_kappa
funcdict['roc_auc'] = roc_auc_score
