"""
Functions to load the dataset.
"""

import pandas as pd
import numpy as np


def load():
    train_file = '../../../numerai_datasets/numerai_training_data.csv'
    test_file = '../../../numerai_datasets/numerai_tournament_data.csv'
    #output_file = '../../../numerai_datasets/predictions_lr.csv'

    #

    train = pd.read_csv( train_file )
    test = pd.read_csv( test_file )

    # no need for validation flag
    train.drop( 'validation', axis = 1 , inplace = True )

    # encode the categorical variable as one-hot, drop the original column afterwards
    # but first let's make sure the values are the same in train and test

    assert( set( train.c1.unique()) == set( test.c1.unique()))

    train_dummies = pd.get_dummies( train.c1 )
    train_num = pd.concat(( train.drop( 'c1', axis = 1 ), train_dummies ), axis = 1 )

    test_dummies = pd.get_dummies( test.c1 )
    test_num = pd.concat(( test.drop( 'c1', axis = 1 ), test_dummies ), axis = 1 )

    #

    y_train = train_num.target.values
    x_train = train_num.drop( 'target', axis = 1 )
    x_test = test_num.drop( 't_id', axis = 1 )
    
    x_train = x_train.as_matrix()
    x_test = x_test.as_matrix()

    return x_train, y_train, x_test

def concat():
    test_file = '../../../numerai_datasets/numerai_tournament_data.csv'
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
    
    concat()
    #X_train, y_train, X_test = load()
