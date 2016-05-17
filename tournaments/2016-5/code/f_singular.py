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

from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedKFold
#import f_select
import warnings
import dill
import os
from numpy import inf
import matplotlib.pyplot as plt
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

def feature_correlations(th=0.05):
    X, Y = load_data(original = True)
    X = X.drop('target', axis=1)
    Y = Y.drop('t_id', axis=1)
    ALL = pd.concat((X,Y))
    Correlations = ALL.corr()
    s = Correlations.stack()
    print s[((s>-th) & (s<th))]
    #for column in Correlations:
    #    for index, row in Correlations[column]:
    #        print index, row
    #plt.matshow(ALL.corr())
    #plt.colorbar()
    #plt.show()


# Find log loss on validation hold out set
def singular_lgls(pcompa = False):
    #X, training_target, Y_test, Y_test_id = load_data()
    X, Y = load_data(original=True)
    
    test_id = Y[['t_id']].as_matrix()
    test_id = test_id.flatten()
   
    training_target = X[['target']].as_matrix()
    training_target = training_target.flatten()
    
    features = []
    lgls = []

    for i in X.columns:
        if str(i) == 'target':
            pass
        else:
            #print "Feature %s " %(str(i))
            features.append(str(i))
            feature_X = X[str(i)]
            feature_Y = Y[str(i)]
            X_np = feature_X.as_matrix() 
            Y_np = feature_Y.as_matrix() 

        # split traininf data in to training and validation set
        X_train, X_Val, train_target, val_target = train_test_split(X_np, training_target, test_size=0.33, random_state=4)
        X_train = np.reshape(X_train, (len(X_train), 1))
        X_Val = np.reshape(X_Val, (len(X_Val), 1))
        np.reshape(train_target, (len(train_target), 1))
        np.reshape(val_target, (len(val_target), 1))

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
            #print j, clf.__class__.__name__
            # pipeline with feature selection, pca and classifier
            if pcompa==True:
                #pipeline = Pipeline([('select', select), ('pca', pca), ('clf', clf)])
                pipeline = Pipeline([('pca', pca), ('clf', clf)])
            else:
                pipeline = Pipeline([('clf', clf)])
                #pipeline = Pipeline([('select', select), ('clf', clf)])

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

            #print 'CV accuracy: %.3f +/- %.3f ' %(
            #                    np.mean(scores), np.std(scores))

            ## test on the hold out set
            #print 'Log Loss: %.5f ' %(log_loss(val_target, pipeline.predict_proba(X_Val)[:, 1]))
            lgls.append(log_loss(val_target, pipeline.predict_proba(X_Val)[:, 1]))

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
    #print sorted(zip(features, lgls), reverse=False, key=lambda x: x[1])[:5]
    print sorted(zip(features, lgls), reverse=False, key=lambda x: x[1])
    print "Average logloss per feature: ", np.mean(lgls)
    return np.mean(lgls)
    #print tabulate(lgls, features, tablefmt="grid")

#differences_lgls()
    #print
    #addition_lgls()
    #print 
    #multiply_lgls()
    #print 
    #
# logloss scores of differences of top 5 singular features from singular_lgls
def combinations_lgls(pcompa = False, differences = True, addition = False, multiplication = False, division = False):
    #X, training_target, Y_test, Y_test_id = load_data()
    X, Y = load_data(original=True)
    
    test_id = Y[['t_id']].as_matrix()
    test_id = test_id.flatten()
   
    training_target = X[['target']].as_matrix()
    training_target = training_target.flatten()
   
    ### INCLUDE ALL NOT JUST THESE 5 ###

    f_s = [ 'feature%d' %x for x in range(1,22)]
    g_s = [ 'feature%d' %x for x in range(1,22)]

    features = []
    lgls = []

    for f in f_s:
        for g in g_s:
            if f == g:
                pass
            else:
                if differences: 
                    features.append(str(f)+"-"+str(g))
                    feature_X = X[str(f)]-X[str(g)]
                    feature_Y = Y[str(f)]-Y[str(g)]
                elif addition:
                    features.append(str(f)+"+"+str(g))
                    feature_X = X[str(f)]+X[str(g)]
                    feature_Y = Y[str(f)]+Y[str(g)]
                elif multiplication:
                    features.append(str(f)+"x"+str(g))
                    feature_X = X[str(f)]*X[str(g)]
                    feature_Y = Y[str(f)]*Y[str(g)]
                elif division:
                    features.append(str(f)+"/"+str(g))
                    feature_X = X[str(f)].div(X[str(g)])
                    feature_Y = Y[str(f)].div(Y[str(g)])

                X_np = feature_X.as_matrix() 
                Y_np = feature_Y.as_matrix() 

                # split traininf data in to training and validation set
                X_train, X_Val, train_target, val_target = train_test_split(X_np, training_target, test_size=0.33, random_state=4)
                X_train = np.reshape(X_train, (len(X_train), 1))
                X_Val = np.reshape(X_Val, (len(X_Val), 1))
                np.reshape(train_target, (len(train_target), 1))
                np.reshape(val_target, (len(val_target), 1))

                # feature selection
                select = SelectKBest(chi2, k=20)

                # dimensionality reduction ( PCA)
                pca = PCA(n_components=2, whiten=True)

                # randomized grid search???

                clfs = [
                        LogisticRegression()]
                        #xgb.XGBClassifier(objective='binary:logistic', max_depth=3, n_estimators=300, learning_rate=0.05),

                for j, clf in enumerate(clfs):
                    #print j, clf.__class__.__name__
                    # pipeline with feature selection, pca and classifier
                    if pcompa==True:
                        #pipeline = Pipeline([('select', select), ('pca', pca), ('clf', clf)])
                        pipeline = Pipeline([('pca', pca), ('clf', clf)])
                    else:
                        pipeline = Pipeline([('clf', clf)])
                        #pipeline = Pipeline([('select', select), ('clf', clf)])

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

                        lgls.append(log_loss(val_target, pipeline.predict_proba(X_Val)[:, 1]))

    combination_scores = sorted(zip(features, lgls), key=lambda x: x[1])
    single_f_average = singular_lgls()
    
    return [x for x in combination_scores if x[1]<single_f_average]

def save_combinations():
         
    div = combinations_lgls(pcompa = False, differences = True, addition = False, multiplication = False, division = False)

if __name__ == '__main__':
    
    #feature_correlations()
    #print
    #singular_lgls()
    #print
    combinations_lgls(pcompa = False, differences = True, addition = False, multiplication = False, division = False)
