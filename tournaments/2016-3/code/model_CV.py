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
from sklearn.cross_validation import StratifiedKFold
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

warnings.filterwarnings("ignore")

np.random.seed(1279)

# ==== Load Data ====

def load_data(f_number=378):
    test_file = '../../../numerai_datasets_new/numerai_tournament_data.csv'
    training_file = '../../../numerai_datasets_new/numerai_training_data.csv'
    feature = 'features__' + str(f_number)  
    #feature = 'features__best_' + str(f_number)

    feature_file = feature + '.pickle'
    if os.path.exists(feature_file):
        with open(feature_file, "rb") as input_file:
            D = dill.load(input_file)
        if os.path.exists(feature + '.txt'):    
            stats = open(feature + '.txt', 'r')
            file_contents = stats.read()
            #print file_contents,
            stats.close()
    else:
        print "Feature set doesn't exist!"

    X               = D[0]
    training_target = D[1]
    Y               = D[2]
    test_id         = D[3]
    #Xall            = D[4]
    
    # split traininf data in to training and validation set
    X_train, X_test, y_train, y_test = train_test_split(X, training_target, test_size=0.33, random_state=4)
    return X, training_target, X_train, X_test, y_train, y_test, Y, test_id


# Find log loss on validation hold out set
def blend_clfs_CV(f_number = 80, pcompa = True, layer = 1, cycles=9):
    if layer == 1:
        X, X_target, X_train, X_Val, train_target, val_target, Y_test, Y_test_id = load_data(f_number=f_number)
    elif layer == 2:
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

        X, X_target, _, _, _, _, Y_test, Y_test_id = load_data(f_number=f_number)
        #newX = np.zeros((X.shape[0], cycles+len(num_clfs)))
        #newY = np.zeros((Y_test.shape[0], cycles+len(num_clfs)))
        
        #for i in range(cycles):
        test_preds     = 'CV_blended_True_layer_1_keras1_feature_' + str(f_number) + '.csv'
        training_preds = 'CV_blended_True_training_layer_1_keras1_feature_' + str(f_number) + '.csv'
        test_preds_df = pd.read_csv(test_preds)
        training_preds_df = pd.read_csv(training_preds)
        test_preds_np = test_preds_df[["probability"]].as_matrix()
        training_preds_np = training_preds_df.as_matrix()
        X = np.concatenate((X, training_preds_np), axis=1)
        Y_test = np.concatenate((Y_test, test_preds_np), axis=1)
        
        for i in range(cycles):
            test_preds     = 'CV_blended_False_layer_1_keras1_' + str(i) + '_feature_' + str(f_number) + '.csv'
            training_preds = 'CV_blended_False_training_layer_1_keras1_' + str(i) + '_feature_' + str(f_number) + '.csv'
            test_preds_df = pd.read_csv(test_preds)
            training_preds_df = pd.read_csv(training_preds)
            test_preds_np = test_preds_df[["probability"]].as_matrix()
            training_preds_np = training_preds_df.as_matrix()
            X = np.concatenate((X, training_preds_np), axis=1)
            Y_test = np.concatenate((Y_test, test_preds_np), axis=1)

        for i,c in enumerate(num_clfs):
            test_preds     = 'CV_layer_1_' + str(c.__class__.__name__) + str(i) + '_feature_' + str(f_number) + '_pca_' + str(pcompa) + '.csv'
            training_preds = 'CV_training_layer_1_' +  str(c.__class__.__name__) + str(i) + '_feature_' + str(f_number) + '_pca_' + str(pcompa) + '.csv'
            test_preds_df = pd.read_csv(test_preds)
            training_preds_df = pd.read_csv(training_preds)
            test_preds_np = test_preds_df[["probability"]].as_matrix()
            training_preds_np = training_preds_df.as_matrix()
            X = np.concatenate((X, training_preds_np), axis=1)
            Y_test = np.concatenate((Y_test, test_preds_np), axis=1)
    
        X_train, X_Val, train_target, val_target = train_test_split(X, X_target, test_size=0.33, random_state=4)
    
    X[X == -inf] = 0
    X_train[X_train == -inf] = 0
    X_Val[X_Val == -inf] = 0
    Y_test[Y_test == -inf] = 0

    #print "Number of total training samples: ", len(X)
    #print "Number of sub-training samples: ", len(X_train)
    #print "Number of validation samples: :", len(X_Val)

    # feature selection
    #select = SelectKBest(chi2, k=7)

    # dimensionality reduction ( PCA)
    pca = PCA(n_components=2, whiten=True)

    # randomized grid search???

    clfs = [
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

    #C_range = 10.0 ** np.arange(-2, 3)
    #gamma_range = 10.0 ** np.arange(-2, 3)
    #param_grid = {"gamma": gamma_range.tolist(), "C": C_range.tolist(), "kernel": ['rbf', 'linear', 'sigmoid', 'poly']}
    #grid = GridSearchCV(SVC(), param_grid, n_jobs=-1, verbose=2)
    #grid = RandomizedSearchCV(SVC(), param_grid, n_iter=20, n_jobs=-1, verbose=2)
    #grid.fit(X, X_target)
    #print("The best classifier is: ", grid.best_estimator_)
    #print(grid.grid_scores_)

    for j, clf in enumerate(clfs):
        print j, clf
        # pipeline with feature selection, pca and classifier
        if pcompa==True:
            #pipeline = Pipeline([('select', select), ('pca', pca), ('clf', clf)])
            pipeline = Pipeline([('pca', pca), ('clf', clf)])
        else:
            pipeline = Pipeline([('clf', clf)])

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
            
            print 'Fold: %s, Class dist: %s, Log loss: %.3f ' %(k+1, np.bincount(train_target[train]), score)

        print 'CV accuracy: %.3f +/- %.3f ' %(
                            np.mean(scores), np.std(scores))

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
        
        #total_training_probabilities
        training_probs = pipeline.predict_proba(X)[:,1]
        training_probs_df = pd.DataFrame(data=training_probs, columns=["probability"])
        training_submission = 'CV_training_layer_' + str(layer) + '_' + str(clf.__class__.__name__) + str(j) + '_feature_' + str(f_number) + '_pca_' + str(pcompa) 
        training_probs_df.to_csv(training_submission + '.csv', index=False)

        ## test on the hold out set
        print 'Log Loss: %.5f ' %(log_loss(val_target, pipeline.predict_proba(X_Val)[:, 1]))
        
        ## test on real test set, save submission
        test_predictions = pipeline.predict_proba(Y_test)[:,1]
        test_predictions_df = pd.DataFrame(data=test_predictions, columns=["probability"])
        Y_test_id.columns = ["t_id"]
        pred_submission = pd.concat((Y_test_id, test_predictions_df), axis = 1)
        submission = 'CV_layer_' + str(layer) + '_' + str(clf.__class__.__name__) + str(j) + '_feature_' + str(f_number)
        pred_submission.to_csv(submission + '.csv', index = False)
        submission_stats = open(submission + '.txt', 'a')
        submission_stats.write(str(clf) + '\n')
        submission_stats.write('pca = ' + str(pcompa) + '\n')
        submission_stats.write('Log Loss on Validation set: %.5f ' %(log_loss(val_target, pipeline.predict_proba(X_Val)[:, 1])) + '\n')
        submission_stats.write(' ' + '\n')
        submission_stats.close()

# Train on all data and create submission files
def blend_clfs(f_number = 5, pcompa = True, layer = 1, cycles = 10):
    
    if layer == 1:
        X, X_target, _, _, _, _, Y, Y_test_id = load_data(f_number=f_number)
    elif layer == 2:
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

        X, X_target, _, _, _, _, Y_test, Y_test_id = load_data(f_number=f_number)
        #newX = np.zeros((X.shape[0], len(num_clfs)))
        #newY = np.zeros((Y_test.shape[0], len(num_clfs)))
        
        #for i,c in enumerate(num_clfs):
        #    test_preds     = 'layer_1_' + str(c.__class__.__name__) + '_feature_' + str(f_number) + '_pca_' + str(pcompa) + '.csv'
        #    test_preds_df = pd.read_csv(test_preds)
        #    test_preds_np = test_preds_df[["probability"]].as_matrix()
        #    if useOldfeatures:
        #        Y_test = np.concatenate((Y_test, test_preds_np), axis=1)
        #    else:
        #        newY[:, i] = test_preds_np[:, -1]
        #training_preds = 'Training_layer_1_feature_' + str(f_number) + '_pca_' + str(pcompa) + '.csv'
        #training_preds_df = pd.read_csv(training_preds)
        #training_preds_np = training_preds_df.as_matrix()
        #if useOldfeatures:
        #    X = np.concatenate((X, training_preds_np), axis=1)
        #else:
        #    newX[:, i] = training_preds_np[:, -1]
     
        #if not useOldfeatures:
        #    X = newX
        test_preds     = 'layer_1_keras1_feature_' + str(f_number) + '.csv'
        test_preds_df = pd.read_csv(test_preds)
        test_preds_np = test_preds_df[["probability"]].as_matrix()
        Y_test = np.concatenate((Y_test, test_preds_np), axis=1)      #    Y = newY
        for i in range(cycles):
            test_preds     = 'layer_1_keras1_' + str(i) + '_feature_' + str(f_number) + '.csv'
            test_preds_df = pd.read_csv(test_preds)
            test_preds_np = test_preds_df[["probability"]].as_matrix()
            Y_test = np.concatenate((Y_test, test_preds_np), axis=1)
        for i,c in enumerate(num_clfs):
            test_preds     = 'layer_1_' + str(c.__class__.__name__) + str(i) + '_feature_' + str(f_number) + '_pca_' + str(pcompa) + '.csv'
            test_preds_df = pd.read_csv(test_preds)
            test_preds_np = test_preds_df[["probability"]].as_matrix()
            Y_test = np.concatenate((Y_test, test_preds_np), axis=1)
        training_preds1 = 'blended_False_keras1_Training_layer_1_feature_' + str(f_number) + '.csv'
        training_preds_df1 = pd.read_csv(training_preds1)
        training_preds_np1 = training_preds_df1.as_matrix()
        training_preds3 = 'blended_True_keras1_Training_layer_1_feature_' + str(f_number) + '.csv'
        training_preds_df3 = pd.read_csv(training_preds3)
        training_preds_np3 = training_preds_df3.as_matrix()
        training_preds2 = 'Training_layer_1_feature_' + str(f_number) + '_pca_' + str(pcompa) + '.csv'
        training_preds_df2 = pd.read_csv(training_preds2)
        training_preds_np2 = training_preds_df2.as_matrix()
        X = np.concatenate((X, training_preds_np1), axis=1)
        X = np.concatenate((X, training_preds_np3), axis=1)
        X = np.concatenate((X, training_preds_np2), axis=1)
        Y = Y_test

    X[X == -inf] = 0
    Y[Y == -inf] = 0
 
    print "Number of total training samples: ", len(X)

    # feature selection
    #select = SelectKBest(chi2, k=7)

    # dimensionality reduction ( PCA)
    pca = PCA(n_components=2, whiten=True)

    # randomized grid search???
    clfs = [
            LogisticRegression(),
            SVC(kernel='rbf', gamma=1.0, C=0.1, probability=True, verbose=True, random_state=1),
            xgb.XGBClassifier(objective='binary:logistic', max_depth=3, n_estimators=300, learning_rate=0.05),
            KNeighborsClassifier(n_neighbors=100),
            RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1, criterion='gini', random_state=1),
            #RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion='entropy', random_state=1)
            RandomForestClassifier(n_estimators=500, max_depth=3, n_jobs=-1, criterion='entropy', random_state=1),
            AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", learning_rate=0.01, n_estimators=50, random_state=1),
            ExtraTreesClassifier(n_estimators=50, max_depth=6, n_jobs=-1, criterion='gini', random_state=1),
            ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_split=5, min_samples_leaf=5, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=50, max_depth=6, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.01, subsample=0.8, loss='exponential', max_depth=6, n_estimators=50)]
       
    ##########
    
    print "Creating train and test sets for blending."

    X_predictions = np.zeros((X.shape[0], len(clfs)))
    Y_predictions = np.zeros((Y.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print j, clf
         # pipeline with feature selection, pca and classifier
        if pcompa==True:
            #pipeline = Pipeline([('select', select), ('pca', pca), ('clf', clf)])
            pipeline = Pipeline([('pca', pca), ('clf', clf)])
        else:
            pipeline = Pipeline([('clf', clf)])

        # cross validation
        # skf = StratifiedKFold(X_target, n_folds=2, random_state=1)
        X_predictions_j = np.zeros((X.shape[0], cycles))
        Y_predictions_j = np.zeros((Y.shape[0], cycles))
        for i in range(cycles):
            print "Cycle ", i
            pipeline.fit(X, X_target)
            train_predictions = pipeline.predict_proba(X)[:,1]
            #print log_loss(y_test, clf.predict(X_test))
            X_predictions_j[:, i] = train_predictions
            Y_predictions_j[:, i] = pipeline.predict_proba(Y)[:,1]
        X_predictions[:,j] = X_predictions_j.mean(1)
        Y_predictions[:,j] = Y_predictions_j.mean(1)
        Y_submission_df = pd.DataFrame(data=Y_predictions[:,j], columns=["probability"])
        Y_test_id.columns = ["t_id"]
        pred_submission = pd.concat((Y_test_id, Y_submission_df), axis = 1)
        submission = 'layer_' + str(layer) + '_' + str(clf.__class__.__name__) + str(j) + '_feature_' + str(f_number) + '_pca_' + str(pcompa)
        pred_submission.to_csv(submission + '.csv', index = False)

    #total_training_probabilities
    training_probs_df = pd.DataFrame(data=X_predictions, columns=["c"+str(j) for j in range(len(clfs))])
    training_submission = 'Training_layer_' + str(layer) + '_feature_' + str(f_number) + '_pca_' + str(pcompa)
    training_probs_df.to_csv(training_submission + '.csv', index=False)
    
    ## test on real test set, save submission
    submission_stats = open(training_submission + '.txt', 'a')
    submission_stats.write(str(clfs) + '\n')
    submission_stats.write('pca = ' + str(pcompa) + '\n')
    submission_stats.write(' ' + '\n')
    submission_stats.close()

if __name__ == '__main__':

    #load_data(f_number=378)
    #blend_clfs_CV(f_number=2)
    for l in [2,14]:
        print l
        blend_clfs_CV(layer=1,f_number=l,pcompa=False)
    for l in [2,14]:
        print l
        blend_clfs(layer=1,f_number=l,pcompa=False)

