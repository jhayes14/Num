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
    X_train, X_Val, train_target, val_target = train_test_split(X_np, training_target, test_size=0.33)
    #X_train, X_Val, train_target, val_target = train_test_split(X_np, training_target, test_size=0.33, random_state=4)

    # feature selection
    select = SelectKBest(chi2, k=20)

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
        skf = StratifiedKFold(train_target, n_folds=5, random_state=1)

        scores = []

        for k, (train, test) in enumerate(skf):
            pipeline.fit(X_train[train], train_target[train])
            if hasattr(pipeline, 'predict_proba'):
                score = log_loss(train_target[test], pipeline.predict_proba(X_train[test])[:, 1])
                print pipeline.predict(X_train[test])[:10], train_target[test][:10]
            else:
                score = log_loss(train_target[test], pipeline.decision_function(X_train[test]))

            scores.append(score)

            #print 'Fold: %s, Class dist: %s, Log loss: %.3f ' %(k+1, np.bincount(train_target[train]), score)

        print 'CV accuracy: %.3f +/- %.3f ' %(
                            np.mean(scores), np.std(scores))

        ## test on the hold out set
        
        print 'Log Loss: %.5f ' %(log_loss(val_target, pipeline.predict_proba(X_Val)[:, 1]))

## UGH!! THIS IS A SIMPLE FORM OF BOOSTING YOU FOOL ##
## USE ADABOOST INSTEAD!! ##

#TODO: Automate the iterations! For iterations > 2 don't add in to validation set from training samples that came from old validation sets.

def boost_train(threshold=0.495):
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

    X_train, X_holdout, train_target, holdout_target = train_test_split(X_np, training_target, test_size=0.33)
    
    X_subtrain, X_val, subtrain_target, val_target = train_test_split(X_train, train_target, test_size=0.33)
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=6, n_jobs=-1, criterion='gini')

    ####### Iteration 1 ##########

    clf.fit(X_subtrain, subtrain_target)

    val_predictions = clf.predict_proba(X_val)
    val_firm_preds = clf.predict(X_val)
   
    #print val_predictions, val_firm_preds

    # how sure the classifier thinks the item is target == 1.
    #val_prob = val_predictions[:,1]

    #misclassified_val = np.array( [ x for i,x in enumerate(X_val) if val_predictions[i] != val_target[i] ] )
    misclassified_val = np.array( [ x for i,x in enumerate(X_val) if ( val_target[i] != val_firm_preds[i] and val_predictions[i][val_target[i]] < threshold ) ] )
    misclassified_val_real_values = np.array( [ val_target[i] for i,x in enumerate(X_val) if ( val_target[i] != val_firm_preds[i] and val_predictions[i][val_target[i]] < threshold ) ] )
    misclassified_val_indices = np.array( [ i for i,x in enumerate(X_val) if ( val_target[i] != val_firm_preds[i] and val_predictions[i][val_target[i]] < threshold ) ] )
    print "Iteration 1"
    print "Classifier misclassified %d out of %d items (according to a threshold of %.3f)" %( len(misclassified_val), len(X_val), threshold)
    print "Mean validation score was %.4f" %(clf.score(X_val, val_target))
    print "Validation set log loss was %.4f" %(log_loss(val_target, clf.predict_proba(X_val)[:,1]))
    print "Mean holdout set score was %.4f" %(clf.score(X_holdout, holdout_target))
    print "Holdout set log loss was %.4f" %(log_loss(holdout_target, clf.predict_proba(X_holdout)[:,1]))
    print

    ###### Iteration 2 ######
    
    #----- Set Up ---------- 

    # add in hard to classify data points from validation set
    TRAIN = zip(X_subtrain, subtrain_target)
    np.random.shuffle(TRAIN)
    X_subtrain, subtrain_target = zip(*TRAIN)
    X_train_remove, X_train_keep = X_subtrain[:len(misclassified_val)], X_subtrain[len(misclassified_val):]
    subtrain_target_remove, subtrain_target_keep = subtrain_target[:len(misclassified_val)], subtrain_target[len(misclassified_val):]
    X_train_keep = np.vstack((X_train_keep, misclassified_val))
    subtrain_target_new = np.concatenate( ( np.array(subtrain_target_keep), misclassified_val_real_values ))
    assert len(X_train_keep) == len(X_subtrain)
    assert len(subtrain_target_new) == len(subtrain_target)
    
    # remove the added in items from validation set and add in new ones
    X_val_new = np.delete(X_val, misclassified_val_indices, axis=0)
    val_target_new = np.delete(np.array(val_target), misclassified_val_indices, axis=0)
    X_val_new = np.vstack((X_val_new, X_train_remove))
    val_target_new = np.concatenate((val_target_new, np.array(subtrain_target_remove)))
    assert len(X_val) == len(X_val_new)
    assert len(val_target_new) == len(val_target)
    
    #-----------------------

    clf.fit(X_train_keep, subtrain_target)

    val_predictions_new = clf.predict_proba(X_val_new)
    val_firm_preds_new = clf.predict(X_val_new)
   
    #print val_predictions, val_firm_preds

    # how sure the classifier thinks the item is target == 1.
    #val_prob = val_predictions[:,1]

    #misclassified_val = np.array( [ x for i,x in enumerate(X_val) if val_predictions[i] != val_target[i] ] )
    misclassified_val_new = np.array( [ x for i,x in enumerate(X_val_new) if ( val_target_new[i] != val_firm_preds_new[i] and val_predictions_new[i][val_target_new[i]] < threshold ) ] )
    misclassified_val_indices_new = np.array( [ i for i,x in enumerate(X_val_new) if ( val_target_new[i] != val_firm_preds_new[i] and val_predictions_new[i][val_target_new[i]] < threshold ) ] )
   
    print "Iteration 2"
    print "Classifier misclassified %d out of %d items (according to a threshold of %.3f)" %( len(misclassified_val_new), len(X_val_new), threshold)
    print "Mean validation score was %.4f" %(clf.score(X_val_new, val_target_new))
    print "Validation set log loss was %.4f" %(log_loss(val_target_new, clf.predict_proba(X_val_new)[:,1]))
    print "Mean holdout set score was %.4f" %(clf.score(X_holdout, holdout_target))
    print "Holdout set log loss was %.4f" %(log_loss(holdout_target, clf.predict_proba(X_holdout)[:,1]))
    print


    #print clf.score(X_val, val_target)
    
    #print 'Log Loss: %.5f ' %(log_loss(holdout_target, clf.predict_proba(X_holdout)[:, 1]))

if __name__ == '__main__':
    
    for i in [0.485]:
        print
        boost_train(threshold=i)
        print



