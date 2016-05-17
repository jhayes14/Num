import copy
from datetime import datetime
import os
import pandas as pd
from scipy.optimize import minimize
from sklearn import cross_validation
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
import re
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb

def models_blend(feature=2, pcompa=False):
    print "#-------------- Models Averaging -----------------#"
    Y = '../../../numerai_datasets_new/numerai_training_data.csv' 
    Y = pd.read_csv(Y)
    y_test = Y.as_matrix(columns=Y.columns[-1:]).flatten()
    
    num_clfs = [
        LogisticRegression(),
        SVC(kernel='rbf', gamma=1.0, C=0.1, probability=True, verbose=True, random_state=1),
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

    predictions = []
    for k, c in enumerate(num_clfs):
        X1 = 'CV_training_layer_' + str(1) + '_' + str(c.__class__.__name__) + str(k) + '_feature_' + str(feature) + '_pca_' + str(pcompa) +'.csv'
        X1_R = 'layer_' + str(1) + '_' + str(c.__class__.__name__) + str(k) + '_feature_' + str(feature) + '_pca_' + str(pcompa) +'.csv'
        X2 = 'CV_training_layer_' + str(2) + '_' + str(c.__class__.__name__) + str(k) + '_feature_' + str(feature) + '_pca_' + str(pcompa) +'.csv'
        X2_R = 'layer_' + str(2) + '_' + str(c.__class__.__name__) + str(k) + '_feature_' + str(feature) + '_pca_' + str(pcompa) +'.csv'
        if os.path.exists(X1) and os.path.exists(X1_R):
            with open(X1) as f:
                lines = (line for line in f if not line.startswith('prob'))
                preds = np.loadtxt(lines)
                print "Score: ", c.__class__.__name__, "layer 1", " -> ", metrics.log_loss(y_test, preds)
                predictions.append(preds)
        else:
            pass
        if os.path.exists(X2) and os.path.exists(X2_R):
            with open(X2) as f:
                lines = (line for line in f if not line.startswith('prob'))
                preds = np.loadtxt(lines)
                print "Score: ", c.__class__.__name__, "layer 2", " -> ", metrics.log_loss(y_test, preds)
                predictions.append(preds)
        else:
            pass
    X = 'CV_blended_True_training_layer_' + str(1) + '_keras1_feature_' + str(feature) + '.csv' 
    X_R = 'layer_' + str(1) + '_keras1_feature_' + str(feature) + '.csv' 
    if os.path.exists(X) and os.path.exists(X_R):
       with open(X) as f:
           lines = (line for line in f if not line.startswith('prob'))
           preds = np.loadtxt(lines)
           print "Score: ", "Keras ", "layer 1", " -> ", metrics.log_loss(y_test, preds)
           #print "Score: ", "Keras ", str(j), " -> ", metrics.log_loss(y_test, preds)
           predictions.append(preds)
    else:
        pass
    X = 'CV_blended_Truetraining_layer_' + str(2) + '_keras1_feature_' + str(feature) + '.csv' 
    X_R = 'layer_' + str(2) + '_keras1_feature_' + str(feature) + '.csv' 
    if os.path.exists(X) and os.path.exists(X_R):
       with open(X) as f:
           lines = (line for line in f if not line.startswith('prob'))
           preds = np.loadtxt(lines)
           print "Score: ", "Keras ", "layer 2", " -> ", metrics.log_loss(y_test, preds)
           #print "Score: ", "Keras ", str(j), " -> ", metrics.log_loss(y_test, preds)
           predictions.append(preds)
    else:
       pass
    for j in range(10):
        X = 'CV_blended_False_training_layer_' + str(1) + '_keras1_' + str(j) + '_feature_' + str(feature) + '.csv' 
        X_R = 'layer_' + str(1) + '_keras1_' + str(j) + '_feature_' + str(feature) + '.csv' 
        if os.path.exists(X) and os.path.exists(X_R):
           with open(X) as f:
               lines = (line for line in f if not line.startswith('prob'))
               preds = np.loadtxt(lines)
               print "Score: ", "Keras ", str(j), " layer 1", " -> ", metrics.log_loss(y_test, preds)
               #print "Score: ", "Keras ", str(j), " -> ", metrics.log_loss(y_test, preds)
               predictions.append(preds)
        else:
           pass
    for j in range(10):
        X = 'CV_blended_False_training_layer_' + str(2) + '_keras1_' + str(j) + '_feature_' + str(feature) + '.csv' 
        X_R = 'layer_' + str(2) + '_keras1_' + str(j) + '_feature_' + str(feature) + '.csv' 
        if os.path.exists(X) and os.path.exists(X_R):
           with open(X) as f:
               lines = (line for line in f if not line.startswith('prob'))
               preds = np.loadtxt(lines)
               print "Score: ", "Keras ", str(j), "layer 2", " -> ", metrics.log_loss(y_test, preds)
               #print "Score: ", "Keras ", str(j), " -> ", metrics.log_loss(y_test, preds)
               predictions.append(preds)
        else:
           pass
   

    def log_loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
            final_prediction += weight * prediction
        return metrics.log_loss(y_test, final_prediction)

    #starting_values = [1/float(len(predictions))] * len(predictions)
    starting_values = [0.5] * len(predictions)
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(predictions)
    #res = minimize(log_loss_func, starting_values, method='BFGS', bounds=bounds, constraints=cons)
    res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)
    print type(res)
    print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))
    return res['x']

def blended_submission(feature=2, pcompa=False):
    best_weights = models_blend(feature=feature)
    #best_weights = np.random.random(21)
    #best_weights /= best_weights.sum()
    #print best_weights.sum()
    print best_weights
    num_clfs = [
        LogisticRegression(),
        SVC(kernel='rbf', gamma=1.0, C=0.1, probability=True, verbose=True, random_state=1),
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

    preds = []
    for k, c in enumerate(num_clfs):
        X1 = 'layer_' + str(1) + '_' + str(c.__class__.__name__) + str(k) + '_feature_' + str(feature) + '_pca_' + str(pcompa) +'.csv'
        X2 = 'layer_' + str(2) + '_' + str(c.__class__.__name__) + str(k) + '_feature_' + str(feature) + '_pca_' + str(pcompa) +'.csv'
        #X = 'CV_training_layer_' + str(layer) + '_' + str(c.__class__.__name__) + '_feature_' + str(feature) + '_pca_' + str(pcompa) +'.csv'
        if os.path.exists(X1):
            print str(c.__class__.__name__), " layer 1" 
            test_preds_df = pd.read_csv(X1)
            test_preds_np = test_preds_df[["probability"]].as_matrix()
            preds.append(test_preds_np)
        else:
            pass
        if os.path.exists(X2):
            print str(c.__class__.__name__), " layer 2" 
            test_preds_df = pd.read_csv(X2)
            test_preds_np = test_preds_df[["probability"]].as_matrix()
            preds.append(test_preds_np)
        else:
            pass
    X = 'layer_' + str(1) + '_keras1_feature_' + str(feature) + '.csv' 
    if os.path.exists(X):
        print "Keras layer 1"
        test_preds_df = pd.read_csv(X)
        test_preds_np = test_preds_df[["probability"]].as_matrix()
        preds.append(test_preds_np)
    else:
        pass
    X = 'layer_' + str(2) + '_keras1_feature_' + str(feature) + '.csv' 
    if os.path.exists(X):
        print "Keras layer 2"
        test_preds_df = pd.read_csv(X)
        test_preds_np = test_preds_df[["probability"]].as_matrix()
        preds.append(test_preds_np)
    else:
        pass
    for j in range(10):
        X = 'layer_' + str(1) + '_keras1_' + str(j) + '_feature_' + str(feature) + '.csv' 
        if os.path.exists(X):
            print "Keras ", str(j), " layer 1"
            test_preds_df = pd.read_csv(X)
            test_preds_np = test_preds_df[["probability"]].as_matrix()
            preds.append(test_preds_np)
        else:
            pass
    for j in range(10):
        X = 'layer_' + str(2) + '_keras1_' + str(j) + '_feature_' + str(feature) + '.csv' 
        if os.path.exists(X):
            print "Keras ", str(j), " layer 2"
            test_preds_df = pd.read_csv(X)
            test_preds_np = test_preds_df[["probability"]].as_matrix()
            preds.append(test_preds_np)
        else:
            pass
    assert len(preds)==len(best_weights)
    preds = [x.flatten() for x in preds]       
    sub = [best_weights[i]*x for i,x in enumerate(preds)]
    sub = sum(sub)
    Y_submission_df = pd.DataFrame(data=sub, columns=["probability"])
    Y_test_id = 'layer_' + str(1) + '_keras1_feature_' + str(feature) + '.csv' 
    Y_id = pd.read_csv(Y_test_id)
    Y_id = test_preds_df[["t_id"]]
    pred_submission = pd.concat((Y_id, Y_submission_df), axis = 1)
    submission = 'blended_predictions_'
    pred_submission.to_csv(submission + '.csv', index = False)
    #return best_weights, preds




if __name__ == '__main__':
    blended_submission(feature=14)
