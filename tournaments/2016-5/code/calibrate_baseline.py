

from __future__ import division
import numpy as np
import pandas as pd
import load_data
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import log_loss
import os
import dill


#TODO: MAKE THIS RUN

X               = D[0]
training_target = D[1]
Y               = D[2]
test_id         = D[3]
#Xall            = D[4]

def blending():
    np.random.seed(0) # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False

    #X, y, X_submission = load_data.load()
    X, y, X_submission = D[0], D[1], D[2]

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    clfs = [
            LogisticRegression(),
            #SVC(kernel='rbf', gamma=0.7, C=1.0, probability=True, verbose=True),
            xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05),
            KNeighborsClassifier(n_neighbors=4),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=50, n_jobs=-1, criterion='entropy'),
            AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
            ExtraTreesClassifier(n_estimators=50, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print "Creating train and test sets for blending."

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            #print log_loss(y_test, clf.predict(X_test))
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
        df = pd.DataFrame({'probability': dataset_blend_test[:,j]})
        header = ['probability']

        #print "saving"
        #df.to_csv('test'+str(j)+'.csv', columns = header, index=False)
    #print dataset_blend_train, dataset_blend_test
    #print len(dataset_blend_train), len(dataset_blend_test)
    #return dataset_blend_train, dataset_blend_test


    print
    print "Blending."
    clf = LogisticRegression()
    clf = RandomForestClassifier(n_estimators=50, n_jobs=1, criterion='gini')
    dataset_blend_train, y, dataset_blend_test = X, y, X_submission
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    df = pd.DataFrame({'probability':y_submission})
    header = ['probability']

    print "Saving Results."
    df.to_csv('../predictions/test_no_calibration.csv', columns = header, index=False)

def calib_blending():
    np.random.seed(0) # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False

    #X, y, X_submission = load_data.load()
    X, y, X_submission = D[0], D[1], D[2]

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    clfs = [
            LogisticRegression(),
            #SVC(kernel='rbf', gamma=0.7, C=1.0, probability=True, verbose=True),
            xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05),
            KNeighborsClassifier(n_neighbors=4),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=50, n_jobs=-1, criterion='entropy'),
            AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
            ExtraTreesClassifier(n_estimators=50, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print "Creating train and test sets for blending."

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print j, clf
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            sig_clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            #print log_loss(y_test, clf.predict(X_test))
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = sig_clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
        df = pd.DataFrame({'probability': dataset_blend_test[:,j]})
        header = ['probability']
        #print "saving"
        #df.to_csv('test'+str(j)+'.csv', columns = header, index=False)
    #print dataset_blend_train, dataset_blend_test
    #print len(dataset_blend_train), len(dataset_blend_test)
    #return dataset_blend_train, dataset_blend_test


    print
    print "Blending."
    clf = LogisticRegression()
    clf = RandomForestClassifier(n_estimators=50, n_jobs=1, criterion='gini')
    dataset_blend_train, y, dataset_blend_test = X, y, X_submission
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    df = pd.DataFrame({'probability':y_submission})
    header = ['probability']

    print "Saving Results."
    df.to_csv('../predictions/test_with_calibration.csv', columns = header, index=False)

blending()
calib_blending()
