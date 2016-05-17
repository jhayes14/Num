import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.decomposition import PCA, FastICA
#import f_blend
from scipy.stats import rankdata
import warnings
import dill
from qsprLib import *
#from interact_analysis import *
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder

def dim_reduce(X, n_components=10):
    pca = PCA(n_components=n_components)
    S_pca_ = pca.fit_transform(X)
    return S_pca_

def add_golden_features(X):
    with_added_features = X
    for i in range(len(X.columns)):
        print i
        for j in range(i+1, len(X.columns)):
            new_col_mult =  preprocessing.scale(np.array(X.ix[:,i]*X.ix[:,j]))
            new_col_plus =  preprocessing.scale(np.array(X.ix[:,i]+X.ix[:,j]))
            with_added_features = np.column_stack((with_added_features, new_col_mult, new_col_plus))
        new_col_log = np.log(preprocessing.minmax_scale(np.array(X.ix[:,i]), feature_range=[1,1000]))
        new_col_rank = preprocessing.scale(np.array(rankdata(X.ix[:,i])))
        with_added_features = np.column_stack((with_added_features, new_col_log, new_col_rank))
    reduced_38 = dim_reduce(X.ix[:, :38])
    with_added_features = np.column_stack((with_added_features, reduced_38))
    print with_added_features.shape
    print len(with_added_features[0])
    return with_added_features

def prepareDataset(seed=123, blended_preds = True, stats=True, combinations = True, createVerticalFeatures=True, logtransform = False, makeDiff=False):

    np.random.seed(seed)

    train_file = '../../../numerai_datasets_new/numerai_training_data.csv'
    test_file = '../../../numerai_datasets_new/numerai_tournament_data.csv'
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    test_id = test['t_id']
    test.drop(['t_id'],axis=1,inplace=True)
    ytrain = train['target']
    train.drop(['target'],axis=1,inplace=True)
    Xall = pd.concat([test, train], ignore_index=True)

    if combinations:
        features = list(Xall.columns)
        for f in features:
            for g in features:
                if f != g:
                    if not (str(g) + "_" + str(f)) in Xall.columns:
                        Xall[str(f) + "_" + str(g)] = Xall[f] * Xall[g]

    if stats:
        Xall['f_mean'] = Xall.mean(axis=1)
        Xall['f_std'] = Xall.std(axis=1)
        Xall['f_var'] = Xall.var(axis=1)
        Xall['f_sem'] = Xall.sem(axis=1)
        Xall['f_mad'] = Xall.mad(axis=1)
        Xall['f_max'] = Xall.max(axis=1)
        Xall['f_min'] = Xall.min(axis=1)
        Xall['f_median'] = Xall.median(axis=1)
        #Xall['f_mode'] = Xall.mode(axis=1)
        Xall['f_prod'] = Xall.prod(axis=1)
        Xall['f_skew'] = Xall.skew(axis=1)
        Xall['f_sum'] = Xall.sum(axis=1)
        #Xall['f_cumsum'] = Xall.cumsum(axis=1)
        #Xall['f_cumpod'] = Xall.cumprod(axis=1)
        #Xall['f_cummax'] = Xall.cummax(axis=1)
        #Xall['f_cummin'] = Xall.cummin(axis=1)
        #Xall['f_compound'] = Xall.compound(axis=1)


    if createVerticalFeatures:
        print "Creating vert features..."
        colnames = list(Xall.columns.values)
        print colnames
        for col in colnames:
                Xall['median_'+col] = 0.0
                Xall['sdev_'+col] = 0.0
                Xall['sum_'+col] = 0.0
                Xall['count_'+col] = 0.0

        print Xall.head()
     
    #if blended_preds:
    #    blend_train, blend_test = f_blend.blending()
    #    blend_all = np.concatenate((blend_test,blend_train))
    #    for i in range(len(blend_all[0])):
    #        Xall[str(i)] = blend_all[:,i]

    # generate polynomial features 
    #Xall = make_polynomials(Xall, Xtest=None, degree=2, cutoff=100,quadratic=True)

    if makeDiff is not None:
        X_diff = differentiateFeatures(Xall.iloc[:,:])
        Xall = pd.concat([Xall, X_diff],axis=1)
        Xall = Xall.ix[:,(Xall != 0).any(axis=0)]

    # second diff??

    # remove useless data ??
    #removeLowVar(Xall, threshhold=1E-5)

    print "Columns used",list(Xall.columns)
    #print Xall["diff11"]
    #split data
    
    
    Xall = add_golden_features(Xall)
    
    Xall = Xall.astype(np.float64)
    Xtrain = Xall[len(test.index):]
    Xtest = Xall[:len(test.index)]
    Xval = None
    yval = None

    #print "Training data:",Xtrain.info()

    train = np.array(Xtrain).astype(np.float32)
    test = np.array(Xtest).astype(np.float32)
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    scaler1.fit(Xtrain)
    scaler2.fit(Xtest)

    training_target = ytrain.values.T.astype(np.int32)

    X = np.array(Xtrain).astype(np.float32)
    X = scaler1.transform(X).astype(np.float32)
    Y = np.array(Xtest).astype(np.float32)
    Y = scaler2.transform(Y).astype(np.float32)
    
    #X = add_golden_features(X)
    #Y = add_golden_features(Y)

    with open("features__3.pickle", "wb") as output_file:
        dill.dump((X, training_target, Y, test_id, Xall), output_file)
    
    return X, training_target, Y, test_id, Xall

prepareDataset()
