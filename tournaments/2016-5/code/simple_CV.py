#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.optimizers import SGD, Adadelta, Adam, rmsprop
#from keras.layers.advanced_activations import LeakyReLU, PReLU
#from keras.layers.normalization import BatchNormalization
#from keras.utils import np_utils, generic_utils
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import KernelPCA, PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedKFold
#import f_select
import warnings
import dill
import os
from numpy import inf

warnings.filterwarnings("ignore")

#np.random.seed(1279)

# ==== Load Data ====

def load_data(num=1, original=False):
    if original:
        training_file = '../../../numerai_datasets_new/numerai_training_data.csv'
        test_file = '../../../numerai_datasets_new/numerai_tournament_data.csv'

        X = pd.read_csv( training_file )
        Y = pd.read_csv( test_file )

    else:
        training_file = '../features/numerai_training_data_' + str(num) + '.csv'
        test_file  = '../features/numerai_tournament_data_' + str(num) + '.csv'

        X = pd.read_csv( training_file, compression='gzip')
        Y = pd.read_csv( test_file, compression='gzip')

    return X, Y


# Find log loss on validation hold out set
def CV_holdout(pcompa = False):
    #X, training_target, Y_test, Y_test_id = load_data()
    X, Y = load_data()

    test_id = Y[['t_id']].as_matrix()
    test_id = test_id.flatten()
    Y = Y.drop( 't_id', axis = 1 )
    training_target = X[['target']].as_matrix()
    training_target = training_target.flatten()
    X = X.drop( 'target', axis = 1)
    X_np = X.as_matrix()
    Y_np = Y.as_matrix()

    # split traininf data in to training and validation set
    X_train, X_Val, train_target, val_target = train_test_split(X_np, training_target, test_size=0.33, random_state=4)

    # feature selection
    select = SelectKBest(chi2, k=20)

    # dimensionality reduction ( PCA)
    pca = PCA(n_components=2, whiten=True)

    # randomized grid search???

    clfs = [
            LogisticRegression(),
            xgb.XGBClassifier(objective='binary:logistic', max_depth=3, n_estimators=300, learning_rate=0.05),
            KNeighborsClassifier(n_neighbors=100),
            RandomForestClassifier(n_estimators=50, max_depth=6, n_jobs=-1, criterion='gini', random_state=1),
            #RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion='entropy', random_state=1)
            RandomForestClassifier(n_estimators=500, max_depth=3, n_jobs=-1, criterion='entropy', random_state=1),
            AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", learning_rate=0.01, n_estimators=50, random_state=1),
            ExtraTreesClassifier(n_estimators=50, max_depth=6, n_jobs=-1, criterion='gini', random_state=1),
            ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_split=5, min_samples_leaf=5, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=50, max_depth=6, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.01, subsample=0.8, loss='exponential', max_depth=6, n_estimators=50)]

    for j, clf in enumerate(clfs):
        print j, clf.__class__.__name__
        # pipeline with feature selection, pca and classifier
        if pcompa==True:
            #pipeline = Pipeline([('select', select), ('pca', pca), ('clf', clf)])
            pipeline = Pipeline([('pca', pca), ('clf', clf)])
        else:
            #pipeline = Pipeline([('clf', clf)])
            pipeline = Pipeline([('select', select), ('clf', clf)])

        # cross validation
        skf = StratifiedKFold(train_target, n_folds=5, random_state=1)

        scores = []

        for k, (train, test) in enumerate(skf):
            pipeline.fit(X_train[train], train_target[train])
            if hasattr(pipeline, 'predict_proba'):
                score = log_loss(train_target[test], pipeline.predict_proba(X_train[test])[:, 1])
            else:
                score = log_loss(train_target[test], pipeline.decision_function(X_train[test]))

            scores.append(score)

            #print 'Fold: %s, Class dist: %s, Log loss: %.3f ' %(k+1, np.bincount(train_target[train]), score)

        print 'CV accuracy: %.3f +/- %.3f ' %(
                            np.mean(scores), np.std(scores))

        ## test on the hold out set
        print 'Log Loss: %.5f ' %(log_loss(val_target, pipeline.predict_proba(X_Val)[:, 1]))

        ## Learning curves
        #train_sizes, train_scores, test_scores = \
        #        learning_curve(estimator=pipeline,
        #                       X=X_train,
        #                       y=train_target,
        #                       train_sizes=np.linspace(.1, 1.0, 5),
        #                       cv=5,
        #                       scoring='log_loss',
        #                       n_jobs=1)

        #train_mean = np.mean(train_scores, axis=1)
        #train_std = np.std(train_scores, axis=1)

        #test_mean = np.mean(test_scores, axis=1)
        #test_std = np.std(test_scores, axis=1)

def CVkfold(pcompa = False):
    #X, training_target, Y_test, Y_test_id = load_data()
    X, Y = load_data()

    test_id = Y[['t_id']].as_matrix()
    test_id = test_id.flatten()
    Y = Y.drop( 't_id', axis = 1 )
    training_target = X[['target']].as_matrix()
    training_target = training_target.flatten()
    X = X.drop( 'target', axis = 1)
    X_np = X.as_matrix()
    Y_np = Y.as_matrix()

    # split traininf data in to training and validation set
    X_train, X_Val, train_target, val_target = train_test_split(X_np, training_target, test_size=0.33, random_state=4)

    # feature selection
    select = SelectKBest(chi2, k=5)

    # dimensionality reduction ( PCA)
    pca = PCA(n_components=2, whiten=True)

    # randomized grid search???

    clfs = [
            LogisticRegression()]
            #xgb.XGBClassifier(objective='binary:logistic', max_depth=3, n_estimators=300, learning_rate=0.05),
            #KNeighborsClassifier(n_neighbors=100),
            #RandomForestClassifier(n_estimators=50, max_depth=6, n_jobs=-1, criterion='gini', random_state=1),
            #RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion='entropy', random_state=1)
            #RandomForestClassifier(n_estimators=500, max_depth=3, n_jobs=-1, criterion='entropy', random_state=1),
            #AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", learning_rate=0.01, n_estimators=50, random_state=1),
            #ExtraTreesClassifier(n_estimators=50, max_depth=6, n_jobs=-1, criterion='gini', random_state=1),
            #ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_split=5, min_samples_leaf=5, n_jobs=-1, criterion='gini'),
            #ExtraTreesClassifier(n_estimators=50, max_depth=6, n_jobs=-1, criterion='entropy'),
            #GradientBoostingClassifier(learning_rate=0.01, subsample=0.8, loss='exponential', max_depth=6, n_estimators=50)]

    for j, clf in enumerate(clfs):
        print j, clf.__class__.__name__
        # pipeline with feature selection, pca and classifier
        if pcompa==True:
            #pipeline = Pipeline([('select', select), ('pca', pca), ('clf', clf)])
            pipeline = Pipeline([('pca', pca), ('clf', clf)])
        else:
            #pipeline = Pipeline([('clf', clf)])
            pipeline = Pipeline([('select', select), ('clf', clf)])

        # cross validation
        #skf = StratifiedKFold(training_target, n_folds=2, shuffle=True, random_state=1)
        skf = KFold(len(training_target), n_folds=5, shuffle=False)

        scores = []

        for k, (train, test) in enumerate(skf):

            pipeline.fit(X_np[train], training_target[train])

            important_features_indexes = select.get_support(indices=True)
            important_features =  [X.iloc[:,i].name for i in important_features_indexes][:-1]
            if hasattr(pipeline, 'predict_proba'):
                score = log_loss(training_target[test], pipeline.predict_proba(X_np[test])[:, 1])
            else:
                score = log_loss(training_target[test], pipeline.decision_function(X_np[test]))

            scores.append(score)

            print 'Fold: %s, Class dist: %s, Log loss: %.3f , Top Features: %s' %(k+1, np.bincount(training_target[train]), score, str(important_features))

        print 'LogLoss : %.9f +/- %.9f ' %(
                            np.mean(scores), np.std(scores))

# Find most important features 
def f_importance(pcompa = False):
    #X, training_target, Y_test, Y_test_id = load_data()
    X, Y = load_data(original=True)

    test_id = Y[['t_id']].as_matrix()
    test_id = test_id.flatten()
    Y = Y.drop( 't_id', axis = 1 )
    training_target = X[['target']].as_matrix()
    training_target = training_target.flatten()
    X = X.drop( 'target', axis = 1)
    X_np = X.as_matrix()
    Y_np = Y.as_matrix()

    # split traininf data in to training and validation set
    X_train, X_Val, train_target, val_target = train_test_split(X_np, training_target, test_size=0.33, random_state=4)

    # feature selection
    select = SelectKBest(chi2, k=20)

    # dimensionality reduction ( PCA)
    pca = PCA(n_components=2, whiten=True)

    # randomized grid search???

    clfs = [
            RandomForestClassifier(n_estimators=500, max_depth=6, n_jobs=-1, criterion='gini', random_state=1)]

    for j, clf in enumerate(clfs):
        print j, clf.__class__.__name__
        # pipeline with feature selection, pca and classifier
        #if pcompa==True:
            #pipeline = Pipeline([('select', select), ('pca', pca), ('clf', clf)])
            #pipeline = Pipeline([('pca', pca), ('clf', clf)])
        #else:
            #pipeline = Pipeline([('clf', clf)])
            #pipeline = Pipeline([('select', select), ('clf', clf)])

        # cross validation
        skf = StratifiedKFold(train_target, n_folds=50, random_state=1)

        total_importance = []

        for k, (train, test) in enumerate(skf):
            clf.fit(X_train[train], train_target[train])
            importances = clf.feature_importances_
            total_importance.append(importances)

        importances = np.mean(total_importance, axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking after 5 iterations:")

        for f in range(X.shape[1]):
                print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

def calibrate_clf(pcompa=False):
    #X, training_target, Y_test, Y_test_id = load_data()
    X, Y = load_data()

    test_id = Y[['t_id']].as_matrix()
    test_id = test_id.flatten()
    Y = Y.drop( 't_id', axis = 1 )
    training_target = X[['target']].as_matrix()
    training_target = training_target.flatten()
    X = X.drop( 'target', axis = 1)
    X_np = X.as_matrix()
    Y_np = Y.as_matrix()

    # split traininf data in to training and validation set
    X_train, X_Val, train_target, val_target = train_test_split(X_np, training_target, test_size=0.33, random_state=4)

    # feature selection
    select = SelectKBest(chi2, k=5)

    # dimensionality reduction ( PCA)
    pca = PCA(n_components=2, whiten=True)

    # randomized grid search???

    clfs = [
            LogisticRegression(),
            #xgb.XGBClassifier(objective='binary:logistic', max_depth=3, n_estimators=300, learning_rate=0.05),
            #KNeighborsClassifier(n_neighbors=100),
            RandomForestClassifier(n_estimators=50, max_depth=6, n_jobs=-1, criterion='gini', random_state=1),
            #RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion='entropy', random_state=1)
            RandomForestClassifier(n_estimators=500, max_depth=3, n_jobs=-1, criterion='entropy', random_state=1),
            #AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", learning_rate=0.01, n_estimators=50, random_state=1),
            #ExtraTreesClassifier(n_estimators=50, max_depth=6, n_jobs=-1, criterion='gini', random_state=1),
            #ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_split=5, min_samples_leaf=5, n_jobs=-1, criterion='gini'),
            #ExtraTreesClassifier(n_estimators=50, max_depth=6, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.01, subsample=0.8, loss='exponential', max_depth=6, n_estimators=50)]

    for j, clf in enumerate(clfs):
        print
        print j, clf.__class__.__name__
        print
        # pipeline with feature selection, pca and classifier
        if pcompa==True:
            #pipeline = Pipeline([('select', select), ('pca', pca), ('clf', clf)])
            pipeline = Pipeline([('pca', pca), ('clf', clf)])
        else:
            #pipeline = Pipeline([('clf', clf)])
            pipeline = Pipeline([('select', select), ('clf', clf)])

        # cross validation
        #skf = StratifiedKFold(training_target, n_folds=2, shuffle=True, random_state=1)
        skf = KFold(len(training_target), n_folds=5, shuffle=False)
 
        #### Uncalibrated ####
        
        print "UNCALIBRATED:"

        scores = []

        for k, (train, test) in enumerate(skf):

            pipeline.fit(X_np[train], training_target[train])

            if hasattr(pipeline, 'predict_proba'):
                score = log_loss(training_target[test], pipeline.predict_proba(X_np[test])[:, 1])
            else:
                score = log_loss(training_target[test], pipeline.decision_function(X_np[test]))

            scores.append(score)

            #print 'Fold: %s, Class dist: %s, Log loss: %.3f ' %(k+1, np.bincount(training_target[train]), score)

        print 'LogLoss : %.9f +/- %.9f ' %(
                            np.mean(scores), np.std(scores))

        #### Calibrated ####

        print 
        print "CALIBRATED:"

        scores = []

        for k, (train, test) in enumerate(skf):
            
            sig_clf = CalibratedClassifierCV(pipeline, method="sigmoid", cv="prefit")

            sig_clf.fit(X_np[train], training_target[train])
            
            if hasattr(sig_clf, 'predict_proba'):
                score = log_loss(training_target[test], sig_clf.predict_proba(X_np[test])[:, 1])
            else:
                score = log_loss(training_target[test], sig_clf.decision_function(X_np[test]))

            scores.append(score)

            #print 'Fold: %s, Class dist: %s, Log loss: %.3f ' %(k+1, np.bincount(training_target[train]), score)

        print 'LogLoss : %.9f +/- %.9f ' %(
                            np.mean(scores), np.std(scores))



if __name__ == '__main__':

    calibrate_clf()
