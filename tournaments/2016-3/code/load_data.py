"""
Functions to load the dataset.
"""

import pandas as pd
import numpy as np


def load():
    train_file = '../../../numerai_datasets_new/numerai_training_data.csv'
    test_file = '../../../numerai_datasets_new/numerai_tournament_data.csv'
    #output_file = '../../../numerai_datasets/predictions_lr.csv'

    #

    train = pd.read_csv( train_file )
    test = pd.read_csv( test_file )

    # encode the categorical variable as one-hot, drop the original column afterwards
    # but first let's make sure the values are the same in train and test

    #assert( set( train.c1.unique()) == set( test.c1.unique()))

    #train_dummies = pd.get_dummies( train.c1 )
    #train_num = pd.concat(( train.drop( 'c1', axis = 1 ), train_dummies ), axis = 1 )

    #test_dummies = pd.get_dummies( test.c1 )
    #test_num = pd.concat(( test.drop( 'c1', axis = 1 ), test_dummies ), axis = 1 )

    #

    y_train = train.target.values
    x_train = train.drop( 'target', axis = 1 )
    x_test = test.drop( 't_id', axis = 1 )

    x_train = x_train.as_matrix()
    x_test = x_test.as_matrix()

    return x_train, y_train, x_test

def blend_preds():
    preds = []

    predsn5 = pd.read_csv("NNout-5-cycles-best.csv") #0.69020
    test_preds_np = predsn5[["probability"]].as_matrix()
    preds.append(test_preds_np)
    predsn2 = pd.read_csv("NNout-2-cycles-best.csv") #0.69046
    test_preds_np = predsn2[["probability"]].as_matrix()
    preds.append(test_preds_np)
    predsn4 = pd.read_csv("NNout-4-cycles-6903.csv") #0.69037
    test_preds_np = predsn4[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds0 = pd.read_csv("New_NNout-3-cycles-1_feature_best.csv") #0.69096
    test_preds_np = preds0[["probability"]].as_matrix()
    preds.append(test_preds_np)
    """preds1 = pd.read_csv("New_NNtest-3-cycles-0_feature.csv")
    test_preds_np = preds1[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds2 = pd.read_csv("New_NNtest-3-cycles-1_feature.csv")
    test_preds_np = preds2[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds3 = pd.read_csv("New_NNtest-3-cycles-2_feature.csv")
    test_preds_np = preds3[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds4 = pd.read_csv("New_NNtest-3-cycles-3_feature.csv")
    test_preds_np = preds4[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds5 = pd.read_csv("New_NNtest-3-cycles-4_feature.csv")
    test_preds_np = preds5[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds6 = pd.read_csv("New_NNtest-3-cycles-5_feature.csv")
    test_preds_np = preds6[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds7 = pd.read_csv("New_NNtest-3-cycles-6_feature.csv")
    test_preds_np = preds7[["probability"]].as_matrix()
    preds.append(test_preds_np)

    preds8 = pd.read_csv("layer_1_GradientBoostingClassifier10_feature_2_pca_False.csv")  # 0.69226
    test_preds_np = preds8[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds9 = pd.read_csv("layer_1_ExtraTreesClassifier9_feature_2_pca_False.csv")  # 0.69226
    test_preds_np = preds9[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds10 = pd.read_csv("layer_1_ExtraTreesClassifier8_feature_2_pca_False.csv")  # 0.69165
    test_preds_np = preds10[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds11 = pd.read_csv("layer_1_ExtraTreesClassifier7_feature_2_pca_False.csv")  # 0.69093
    test_preds_np = preds11[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds12 = pd.read_csv("layer_1_AdaBoostClassifier6_feature_2_pca_False.csv")  # 0.69299
    test_preds_np = preds12[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds13 = pd.read_csv("layer_1_RandomForestClassifier5_feature_2_pca_False.csv")  # 0.69234
    test_preds_np = preds13[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds14 = pd.read_csv("layer_1_RandomForestClassifier4_feature_2_pca_False.csv")  # 0.69108
    test_preds_np = preds14[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds15 = pd.read_csv("layer_1_KNeighborsClassifier3_feature_2_pca_False.csv")  # 0.69202
    test_preds_np = preds15[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds16 = pd.read_csv("layer_1_XGBClassifier2_feature_2_pca_False.csv")  # 0.69202
    test_preds_np = preds16[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds17 = pd.read_csv("layer_1_SVC1_feature_2_pca_False.csv")  # 0.69202
    test_preds_np = preds17[["probability"]].as_matrix()
    preds.append(test_preds_np)
    preds18 = pd.read_csv("layer_1_LogisticRegression0_feature_2_pca_False.csv")  # 0.69202
    test_preds_np = preds18[["probability"]].as_matrix()
    preds.append(test_preds_np)"""

    #preds = 0.02*preds0 + 0.02*preds1 + 0.8*preds2 + 0.02*preds3 + 0.02*preds4 + \
    #        0.1*preds5 + 0.02*preds6
    print len(preds)
    four_weights = [0.79, 0.06, 0.12, 0.03]
    #last_weights = np.random.uniform(size=18)
    #last_weights /= last_weights.sum()
    #last_weights /= 10
    #four_weights.extend(last_weights)
    print four_weights
    print sum(four_weights)

    sub = [four_weights[i]*x for i,x in enumerate(preds)]
    sub = sum(sub)
    Y_submission_df = pd.DataFrame(data=sub, columns=["probability"])
    Y_test_id = 'layer_1_LogisticRegression0_feature_2_pca_False.csv'
    Y_id = pd.read_csv(Y_test_id)
    Y_id = Y_id[["t_id"]]
    pred_submission = pd.concat((Y_id, Y_submission_df), axis = 1)
    submission = 'blended_predictions_'
    pred_submission.to_csv(submission + '.csv', index = False)
    #preds = 0.77*predsn5 + 0.09*predsn2 + 0.12*predsn4 + 0.019*preds0 + \
    #        0.000142857142*preds1 + 0.000142857142*preds2 + 0.000142857142*preds3 + 0.000142857142*preds4 + \
    #        0.000142857142*preds5 + 0.000142857142*preds6 + 0.000142857142*preds7 #+ 0.028*preds8 + \
    #        0.033*predsb

    #test_file = '../../../numerai_datasets_new/numerai_tournament_data.csv'
    #test = pd.read_csv( test_file )
    #test_ids = test[['t_id']]
    #total = pd.concat((test_ids, preds['probability']), axis = 1)
    #total.to_csv('blended_preds.csv', index=False)

def concat():
    test_file = '../../../numerai_datasets_new/numerai_tournament_data.csv'
    preds = 'test.csv'
    test = pd.read_csv( test_file )
    preds = pd.read_csv( preds )
    test_ids = test[['t_id']]
    #print preds
    #test_ids = np.array(test_ids)
    print preds.iloc[:10]
    print test_ids.iloc[0]
    total = pd.concat(( test_ids, preds), axis = 1)
    #print total.iloc[:10]
    total.to_csv('out.csv', index=False)

if __name__ == '__main__':

    #concat()
    blend_preds()
    #X_train, y_train, X_test = load()
