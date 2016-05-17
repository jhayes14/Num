"""Kaggle competition: Predicting a Biological Response.

Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. The blending scheme is related to the idea Jose H. Solorzano
presented here:
http://www.kaggle.com/c/bioresponse/forums/t/1889/question-about-the-process-of-ensemble-learning/10950#post10950
'''You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''. Or at least this is the
implementation of my understanding of that idea :-)

The predictions are saved in test.csv. The code below created my best
submission to the competition:
- public score (25%): 0.43464
- private score (75%): 0.37751
- final rank on the private leaderboard: 17th over 711 teams :-)

Note: if you increase the number of estimators of the classifiers,
e.g. n_estimators=1000, you get a better score/rank on the private
test set.

Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.
"""

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
from sklearn.svm import SVC
from sklearn.metrics import log_loss
import os
import dill

def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))

def DNN():
    num_features = len(X[0])
    params = {'units1': 861.9044685307921, 'units3': 1006.6951889338304, 'units2': 251.89272280392626, 'optimizer': 'adadelta', 'dropout3': 0.7343552104701996, 'batch_size': 1000, 'num_layers': 2, 'nb_epochs': 20, 'dropout2': 0.26157972777890454, 'dropout1': 0.5597561948479193, 'activation': 'relu'}   
        
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim = num_features)) 
    model.add(Activation(params['activation']))
    #model.add(BatchNormalization())
    model.add(Dropout(params['dropout1']))
    model.add(Dense(output_dim=params['units2'], init = "glorot_uniform")) 
    model.add(Activation(params['activation']))
    #model.add(BatchNormalization())
    model.add(Dropout(params['dropout2']))
    model.add(Dense(output_dim=params['units3'], init = "glorot_uniform")) 
    model.add(Activation(params['activation']))
    #model.add(BatchNormalization())
    model.add(Dropout(params['dropout2']))    
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')

test_file = '../../../numerai_datasets_new/numerai_tournament_data.csv'
training_file = '../../../numerai_datasets_new/numerai_training_data.csv'
feature_file = 'features_wo_preds_best.pickle'
if os.path.exists(feature_file):
    with open(feature_file, "rb") as input_file:
        D = dill.load(input_file)
else:
    D = f_select.prepareDataset()

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
        print "saving"
        df.to_csv('test'+str(j)+'.csv', columns = header, index=False)
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
    df.to_csv('test.csv', columns = header, index=False)

blending()
