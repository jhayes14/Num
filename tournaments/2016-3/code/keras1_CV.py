from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adadelta, Adam, rmsprop, Adamax
from keras.layers.advanced_activations import LeakyReLU, PReLU, SReLU, ThresholdedReLU, ThresholdedLinear, ParametricSoftplus, ELU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils, generic_utils
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedKFold
#import f_select
import warnings
import dill
import os
import model_CV
from numpy import inf

warnings.filterwarnings("ignore")

np.random.seed(1279)


# ==== Load Data from model_CV.py ====

# ==== Parameters ====

#num_features = X.shape[1]
#cycles = 3 

# parameters for DNN and turn targets to categorical targets
def parameters(n=3):
    params = {'units1': [1024, 1024, 1024]*n, 'units2': [2048, 1024, 2048]*n, \
           'units3': [1024, 2048, 2048]*n, 'optimizer': ['adam','adam', 'adam', 'sgd', 'sgd', 'sgd', 'adamax', 'adamax', 'adamax'], \
           'dropout1' : [0.2, 0.2, 0.8]*n, 'dropout2' : [0.8, 0.8, 0.2]*n, \
           'dropout3': [0.8, 0.2, 0.5]*n, 'batch_size': [1000, 1000, 1000]*n, \
           'nb_epochs': [5, 5, 5]*n, 'activation': ['relu','relu','relu']*n, \
           'learning_rate' : [0.2], 'decay' : [1e-6], 'momentum' : [0.9]}
    return params

#def parameters():
#    params = {'units1': [1024], 'units2': [1024], \
#           'units3': [2048], \
#           'dropout1' : [0.8], 'dropout2' : [0.2], \
#           'dropout3': [0.5], 'batch_size': [1000], \
#           'nb_epochs': [3], 'activation': ['relu'], \
#           'learning_rate' : [0.1],  'decay_rate' : [1e-5], 'momentum' : [0.8]} 
#    return params



#--------------#

#val_auc = np.zeros(cycles)

# ==== Defining the neural network model/shape ====

def build_modelA(params, num_features, i):
    modelA = Sequential()
    modelA.add(Dense(output_dim=params['units1'][i], input_dim = num_features)) 
    modelA.add(Activation(params['activation'][i]))
    modelA.add(LeakyReLU())
    #modelA.add(BatchNormalization())
    modelA.add(Dropout(params['dropout1'][i]))
    modelA.add(Dense(output_dim=params['units2'][i], init = "glorot_uniform")) 
    modelA.add(Activation(params['activation'][i]))
    modelA.add(PReLU())
    #modelA.add(BatchNormalization())
    modelA.add(Dropout(params['dropout2'][i]))
    modelA.add(Dense(output_dim=params['units3'][i], init = "glorot_uniform")) 
    modelA.add(Activation(params['activation'][i]))
    modelA.add(PReLU())
    #modelA.add(BatchNormalization())
    modelA.add(Dropout(params['dropout3'][i]))    
    modelA.add(Dense(2))
    modelA.add(Activation('sigmoid'))
    
    if params['optimizer'][i] == 'sgd':
        sgd = SGD(lr=params['learning_rate'][0], decay=params['decay'][0], momentum=params['momentum'][0], nesterov=True)
        modelA.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')
    else:
        modelA.compile(loss='binary_crossentropy', optimizer=params['optimizer'][i], class_mode='binary')
    return modelA

# Find log loss on validation hold out set
def modelA_CV(f_number = 2, layer = 1, pcompa = False, iterates=1, cycles=9, blended=False):
    
    params = parameters()

    if layer == 1:
        X, X_target, X_train, X_Val, train_target, val_target, Y_test, Y_test_id = model_CV.load_data(f_number=f_number)
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
        
        X, X_target, _, _, _, _, Y_test, Y_test_id = model_CV.load_data(f_number=f_number)
        #newX = np.zeros((X.shape[0], cycles+len(num_clfs)))
        #newY = np.zeros((Y_test.shape[0], cycles+len(num_clfs)))
        #for i in range(cycles):
        test_preds     = 'CV_layer_1_keras1_feature_' + str(f_number) + '.csv'
        training_preds = 'CV_blended_True_training_layer_1_keras1_feature_' + str(f_number) + '.csv'
        test_preds_df = pd.read_csv(test_preds)
        training_preds_df = pd.read_csv(training_preds)
        test_preds_np = test_preds_df[["probability"]].as_matrix()
        training_preds_np = training_preds_df.as_matrix()
        X = np.concatenate((X, training_preds_np), axis=1)
        Y_test = np.concatenate((Y_test, test_preds_np), axis=1)
        for i in range(cycles):
            test_preds     = 'CV_layer_1_keras1_' + str(i) + '_feature_' + str(f_number) + '.csv'
            training_predsF = 'CV_blended_False_training_layer_1_keras1_' + str(i) + '_feature_' + str(f_number) + '.csv'
            test_preds_df = pd.read_csv(test_preds)
            training_preds_dfF = pd.read_csv(training_predsF)
            test_preds_np = test_preds_df[["probability"]].as_matrix()
            training_preds_npF = training_preds_dfF.as_matrix()
            X = np.concatenate((X, training_preds_npF), axis=1)
            Y_test = np.concatenate((Y_test, test_preds_np), axis=1)
        
        for i,c in enumerate(num_clfs):
            test_preds     = 'CV_layer_1_' + str(c.__class__.__name__) + '_feature_' + str(f_number) + '_pca_' + str(pcompa) + '.csv'
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

    cat_train_target = np_utils.to_categorical(train_target, 2)
    cat_val_target   = np_utils.to_categorical(val_target, 2)
    num_features = len(X_train[0])


    print "Number of total training samples: ", len(X)
    print "Number of sub-training samples: ", len(X_train)
    print "Number of validation samples: :", len(X_Val)
    
    # randomized grid search???
    if not blended:
        clfs = [
                build_modelA(params, num_features, i=0),
                build_modelA(params, num_features, i=1),
                build_modelA(params, num_features, i=2),
                build_modelA(params, num_features, i=3),
                build_modelA(params, num_features, i=4),
                build_modelA(params, num_features, i=5),
                build_modelA(params, num_features, i=6),
                build_modelA(params, num_features, i=7),
                build_modelA(params, num_features, i=8)]

        for j, clf in enumerate(clfs):
            print j, clf
            scores = []
            
            for i in range(iterates):
                clf.fit(X_train, cat_train_target, batch_size=params['batch_size'][j], nb_epoch=params['nb_epochs'][j], verbose=1)
                valid_preds = clf.predict_proba(X_Val, batch_size=params['batch_size'][j], verbose=1)[:,1]
                if hasattr(clf, 'predict_proba'):
                    score = log_loss(cat_val_target, valid_preds)
                else:
                    score = log_loss(cat_val_target, valid_preds)
                scores.append(score)
                
                #print 'Cycle: %s, Log loss: %.3f ' %(i+1, score)

            #print 'CV accuracy: %.3f +/- %.3f ' %(
            #                    np.mean(scores), np.std(scores))

            #total_training_probabilities
            training_probs = clf.predict_proba(X)[:,1]
            training_probs_df = pd.DataFrame(data=training_probs, columns=["probability"])
            training_submission = 'CV_blended_' + str(blended) + '_training_layer_' + str(layer) + '_keras1_' + str(j) + '_feature_' + str(f_number)
            training_probs_df.to_csv(training_submission + '.csv', index=False)

            ## test on the hold out set
            print 'Log Loss: %.5f ' %(log_loss(cat_val_target, clf.predict_proba(X_Val)[:, 1]))
            print
           
            test_predictions = clf.predict_proba(Y_test)[:,1]
            test_predictions_df = pd.DataFrame(data=test_predictions, columns=["probability"])
            Y_test_id.columns = ["t_id"]
            pred_submission = pd.concat((Y_test_id, test_predictions_df), axis = 1)
            submission  = 'CV_blended_False_layer_1_keras1_' + str(j) + '_feature_' + str(f_number) + '.csv'
            pred_submission.to_csv(submission + '.csv', index = False)
            submission_stats = open(submission + '.txt', 'a')
            submission_stats.write(str(clf) + '\n')
            submission_stats.write('pca = ' + str(pcompa) + '\n')
            submission_stats.write('Log Loss on Validation set: %.5f ' %(log_loss(cat_val_target, valid_preds)) + '\n')
            submission_stats.write(' ' + '\n')
            submission_stats.close()
    else: 
        X_Val_predictions_j = np.zeros((X_Val.shape[0], cycles))
        X_predictions_j = np.zeros((X.shape[0], cycles))

        for i in range(cycles):
            print "Cycle ", i
            model = build_modelA(params, num_features, i=i)
            model.fit(X_train, cat_train_target, batch_size=params['batch_size'][i], nb_epoch=params['nb_epochs'][i], verbose=1)
            valid_preds = model.predict_proba(X_Val, batch_size=params['batch_size'][i], verbose=1)[:,1]
            X_Val_predictions_j[:, i] = valid_preds
            train_predictions = model.predict_proba(X)[:,1]
            X_predictions_j[:, i] = train_predictions 
        
        X_Val_predictions = X_Val_predictions_j.mean(1)
        X_predictions = X_predictions_j.mean(1)
       
        training_probs_df = pd.DataFrame(data=X_predictions, columns=["probability"])
        training_submission = 'CV_blended_' + str(blended) + '_training_layer_' + str(layer) + '_keras1_feature_' + str(f_number)
        training_probs_df.to_csv(training_submission + '.csv', index=False)

            ## test on the hold out set
        print 'Log Loss: %.5f ' %(log_loss(cat_val_target, X_Val_predictions))
        print
        test_predictions = model.predict_proba(Y_test)[:,1]
        test_predictions_df = pd.DataFrame(data=test_predictions, columns=["probability"])
        Y_test_id.columns = ["t_id"]
        pred_submission = pd.concat((Y_test_id, test_predictions_df), axis = 1)
        submission  = 'CV_blended_True_layer_1_keras1_feature_' + str(f_number) 
        pred_submission.to_csv(submission + '.csv', index = False)
        submission_stats = open(submission + '.txt', 'a')
        submission_stats.write('pca = ' + str(pcompa) + '\n')
        submission_stats.write('Log Loss on Validation set: %.5f ' %(log_loss(cat_val_target, X_Val_predictions)) + '\n')
        submission_stats.write(' ' + '\n')
        submission_stats.close()

# Find log loss on validation hold out set
def modelA(f_number = 2, layer = 1, pcompa = False, cycles = 9, blended=False):
    
    params = parameters()

    if layer == 1:
        X, X_target, _, _, _, _, Y, Y_test_id = model_CV.load_data(f_number=f_number)
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
        
        X, X_target, _, _, _, _, Y_test, Y_test_id = model_CV.load_data(f_number=f_number)
        #newX = np.zeros((X.shape[0], cycles+len(num_clfs)))
        #newY = np.zeros((Y_test.shape[0], cycles+len(num_clfs)))
        #for i in range(cycles):
        test_preds     = 'layer_1_keras1_feature_' + str(f_number) + '.csv'
        test_preds_df = pd.read_csv(test_preds)
        test_preds_np = test_preds_df[["probability"]].as_matrix()
        Y_test = np.concatenate((Y_test, test_preds_np), axis=1)
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
        training_preds1 = 'blended_' + str(False) + '_keras1_Training_layer_' + str(layer) + '_feature_' + str(f_number)
        training_preds_df1 = pd.read_csv(training_preds1)
        training_preds_np1 = training_preds_df1.as_matrix()
        training_preds2 = 'blended_' + str(True) + '_keras1_Training_layer_' + str(layer) + '_feature_' + str(f_number)
        training_preds_df2 = pd.read_csv(training_preds2)
        training_preds_np2 = training_preds_df2.as_matrix()
        training_preds3 = 'Training_layer_1_feature_' + str(f_number) + '_pca_' + str(pcompa) + '.csv'
        training_preds_df3 = pd.read_csv(training_preds3)
        training_preds_np3 = training_preds_df3.as_matrix()
        X = np.concatenate((X, training_preds_np2), axis=1)
        X = np.concatenate((X, training_preds_np1), axis=1)
        X = np.concatenate((X, training_preds_np3), axis=1)
        Y = Y_test

    X[X == -inf] = 0
    Y[Y == -inf] = 0

    cat_X_target = np_utils.to_categorical(X_target, 2)
    num_features = len(X[0])


    print "Number of total training samples: ", len(X)
    
    # randomized grid search???
    if not blended:
        clfs = [
                build_modelA(params, num_features, i=0),
                build_modelA(params, num_features, i=1),
                build_modelA(params, num_features, i=2),
                build_modelA(params, num_features, i=3),
                build_modelA(params, num_features, i=4),
                build_modelA(params, num_features, i=5),
                build_modelA(params, num_features, i=6),
                build_modelA(params, num_features, i=7),
                build_modelA(params, num_features, i=8)]

        X_predictions = np.zeros((X.shape[0], len(clfs)))
        Y_predictions = np.zeros((Y.shape[0], len(clfs)))

        for j, clf in enumerate(clfs):
            print j, clf
            
            X_predictions_j = np.zeros((X.shape[0], cycles))
            Y_predictions_j = np.zeros((Y.shape[0], cycles))
           
            for i in range(cycles):
                print "Cycle ", i
                clf.fit(X, cat_X_target, batch_size=params['batch_size'][j], nb_epoch=params['nb_epochs'][j], verbose=1)
                train_predictions = clf.predict_proba(X)[:,1]
                X_predictions_j[:, i] = train_predictions
                test_preds = clf.predict_proba(Y, batch_size=params['batch_size'][j], verbose=1)[:,1]
                Y_predictions_j[:, i] = test_preds
            
            X_predictions[:,j] = X_predictions_j.mean(1)
            Y_predictions[:,j] = Y_predictions_j.mean(1)
            Y_submission_df = pd.DataFrame(data=Y_predictions[:,j], columns=["probability"])
            Y_test_id.columns = ["t_id"]
            pred_submission = pd.concat((Y_test_id, Y_submission_df), axis = 1)
            submission = 'layer_' + str(layer) + '_keras1_' + str(j) + '_feature_' + str(f_number)
            pred_submission.to_csv(submission + '.csv', index = False)

    else:
        Y_predictions_j = np.zeros((Y.shape[0], cycles))
        X_predictions_j = np.zeros((X.shape[0], cycles))

        for i in range(cycles):
            print "Cycle ", i
            model = build_modelA(params, num_features, i=i)
            model.fit(X, cat_X_target, batch_size=params['batch_size'][i], nb_epoch=params['nb_epochs'][i], verbose=1)
            test_preds = model.predict_proba(Y, batch_size=params['batch_size'][i], verbose=1)[:,1]
            Y_predictions_j[:, i] = test_preds
            train_predictions = model.predict_proba(X)[:,1]
            X_predictions_j[:, i] = train_predictions 
        
        X_predictions = X_predictions_j.mean(1)
        Y_predictions = Y_predictions_j.mean(1)
        Y_submission_df = pd.DataFrame(data=Y_predictions, columns=["probability"])
        #Y_submission_df = pd.DataFrame(data=Y_predictions[:,j], columns=["probability"])
        Y_test_id.columns = ["t_id"]
        pred_submission = pd.concat((Y_test_id, Y_submission_df), axis = 1)
        #submission = 'layer_' + str(layer) + '_keras1_' + str(j) + '_feature_' + str(f_number)
        submission = 'layer_' + str(layer) + '_keras1_feature_' + str(f_number)
        pred_submission.to_csv(submission + '.csv', index = False)
    
    #total_training_probabilities
    training_probs_df = pd.DataFrame(data=X_predictions, columns=["c"])
    training_submission = 'blended_' + str(blended) + '_keras1_Training_layer_' + str(layer) + '_feature_' + str(f_number)
    training_probs_df.to_csv(training_submission + '.csv', index=False)
    
    ## test on real test set, save submission
    submission_stats = open(training_submission + '.txt', 'a')
    submission_stats.write(str(cycles) + '\n')
    submission_stats.write('Total params = ' + str(params) + '\n')
    submission_stats.write(' ' + '\n')
    submission_stats.close()

       # ==== Create submission ====

if __name__ == '__main__':

    #load_data(f_number=378)
    #blend_clfs_CV(f_number=2)
    #for l in [2]:
    #    print l
    #    print
    #    modelA_CV(f_number=l, blended=False)
    #    modelA_CV(f_number=l, blended=True)
    #for l in [14]:
    #    print l
    #    print
    #    modelA_CV(f_number=l, blended=False)
    #    modelA_CV(f_number=l, blended=True)

    for l in [2,14]:
        print l
        print
        #modelA(f_number=l, blended=False)
        modelA(f_number=l, blended=True)
