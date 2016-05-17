import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.decomposition import PCA, FastICA
from scipy.stats import rankdata
import warnings
import dill
from qsprLib import *
#from interact_analysis import *
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, Normalizer
from sys import stdout
import os

def dim_reduce(X, n_components=10):
    pca = PCA(n_components=n_components)
    S_pca_ = pca.fit_transform(X)
    return S_pca_

def add_golden_features(X):
    with_added_features = X
    for i in range(len(X.columns)):
        stdout.write("\r%d out of %d" %(i,len(X.columns)))
        stdout.flush()
        for j in range(i+1, len(X.columns)):
            new_col_mult =  preprocessing.scale(np.array(X.ix[:,i]*X.ix[:,j]))
            new_col_plus =  preprocessing.scale(np.array(X.ix[:,i]+X.ix[:,j]))
            with_added_features = np.column_stack((with_added_features, new_col_mult, new_col_plus))
        new_col_log = np.log(preprocessing.minmax_scale(np.array(X.ix[:,i]), feature_range=(0,1)))
        new_col_rank = preprocessing.scale(np.array(rankdata(X.ix[:,i])))
        with_added_features = np.column_stack((with_added_features, new_col_log, new_col_rank))
    reduced_38 = dim_reduce(X.ix[:, :38])
    with_added_features = np.column_stack((with_added_features, reduced_38))
    stdout.write("\n")
    s0, s1 = with_added_features.shape
    columns = ['gf_'+str(j) for j in range(s1)]
    df2 = pd.DataFrame(with_added_features, columns=columns)
    return df2


def prepareDataset(seed=123, gen_polys1 = True,
        stats1 = False, blended_preds = False,
        firDiff = False, secDiff = False,
        stats2 = False, gen_polys2 = False,
        removeUseless = False, Log = False, goldenFeatures = False,
        Scaling = False, num=1):

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

    print Xall.head()

    if gen_polys1:
        print "Adding polynomial1 features..."
        # generate polynomial features
        Xpoly = make_polynomials(Xall, Xtest=None, \
                degree=2, cutoff=100,quadratic=True)

        Xall = pd.concat([Xall, Xpoly], ignore_index=False, axis=1, join='inner')


    if stats1:
        print "Adding stats1..."
        Xall['f1_mean'] = Xall.mean(axis=1)
        Xall['f1_std'] = Xall.std(axis=1)
        Xall['f1_var'] = Xall.var(axis=1)
        Xall['f1_sem'] = Xall.sem(axis=1)
        Xall['f1_mad'] = Xall.mad(axis=1)
        Xall['f1_max'] = Xall.max(axis=1)
        Xall['f1_min'] = Xall.min(axis=1)
        Xall['f1_median'] = Xall.median(axis=1)
        Xall['f1_prod'] = Xall.prod(axis=1)
        Xall['f1_skew'] = Xall.skew(axis=1)
        Xall['f1_sum'] = Xall.sum(axis=1)
        Xall['f1_kurt'] = Xall.kurt(axis=1)
        Xall['f1_skew'] = Xall.skew(axis=1)

    #showAVGCorrelations(Xall)
    #print Xall.tail()

    """if blended_preds:
        print "Adding blended preds as features..."
        blend_train, blend_test = f_blend.blending()
        blend_all = np.concatenate((blend_test,blend_train))
        for i in range(len(blend_all[0])):
            Xall[str(i)] = blend_all[:,i]"""


    if firDiff:
        print "Adding 1st order differential features..."
        X_diff = differentiateFeatures(Xall.iloc[:,:])
        Xall = pd.concat([Xall, X_diff],ignore_index=False, axis=1, join='inner')
        Xall = Xall.ix[:,(Xall != 0).any(axis=0)]


    if secDiff:
        print "Adding 2nd order differential features..."
        X_diff = differentiateFeatures(Xall.iloc[:,:])
        Xall = pd.concat([Xall, X_diff], ignore_index=False, axis=1, join='inner')
        Xall = Xall.ix[:,(Xall != 0).any(axis=0)]


    if stats2:
        print "Adding stats2..."
        Xall['f2_mean'] = Xall.mean(axis=1)
        Xall['f2_std'] = Xall.std(axis=1)
        Xall['f2_var'] = Xall.var(axis=1)
        Xall['f2_sem'] = Xall.sem(axis=1)
        Xall['f2_mad'] = Xall.mad(axis=1)
        Xall['f2_max'] = Xall.max(axis=1)
        Xall['f2_min'] = Xall.min(axis=1)
        Xall['f2_median'] = Xall.median(axis=1)
        #Xall['f2_mode'] = Xall.mode(axis=1)
        Xall['f2_prod'] = Xall.prod(axis=1)
        Xall['f2_skew'] = Xall.skew(axis=1)
        Xall['f2_sum'] = Xall.sum(axis=1)
        #Xall['f2_cumsum'] = Xall.cumsum(axis=1)
        #Xall['f2_cumpod'] = Xall.cumprod(axis=1)
        #Xall['f2_cummax'] = Xall.cummax(axis=1)
        #Xall['f2_cummin'] = Xall.cummin(axis=1)
        #Xall['f2_compound'] = Xall.compound(axis=1)

    if gen_polys2:
        print "Adding polynomial2 features..."
        # generate polynomial features
        Xpoly2 = make_polynomials(Xall, Xtest=None, \
                degree=2, cutoff=100,quadratic=True)

        Xall = pd.concat([Xall, Xpoly2], ignore_index=False, axis=1, join='inner')


    if removeUseless:
        print "Removing low variance features..."
        # remove useless data ??
        removeLowVar(Xall, threshhold=0.1)

    if Log:
        print "Adding log tranfsorms..."
        features = list(Xall.columns)
        for i, f in enumerate(features):
            stdout.write("\r%d out of %d" %(i,len(features)))
            stdout.flush()
            Xall[str(f) + "log"] = np.log1p(Xall[f])
        Xall.fillna(0, inplace=True)
        stdout.write("\n")

    if goldenFeatures:
        print "Adding golden features function..."
        Xgf = add_golden_features(Xall)
        Xall = pd.concat([Xall, Xgf], ignore_index=False, axis=1, join='inner')


    if Scaling:
        Xall.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

    Xtrain = Xall[len(test.index):]
    Xtest = Xall[:len(test.index)]


    Xtest = pd.concat([test_id, Xtest], axis=1)
    Xtrain = pd.concat([Xtrain, ytrain], axis=1, join='inner')


    new_train_file = '../features/numerai_training_data_' + str(num) + '.csv'
    new_test_file  = '../features/numerai_tournament_data_' + str(num) + '.csv'

    if os.path.exists(new_train_file):
        os.remove(new_train_file)

    if os.path.exists(new_test_file):
        os.remove(new_test_file)

    #with gzip.open('path_to_file', 'wt') as write_file:
    #    data_frame.to_csv(write_file)
    print "Creating training file..."
    Xtrain.to_csv(new_train_file, index=False, columns = Xtrain.columns, compression='gzip')
    print "Creating test file..."
    Xtest.to_csv(new_test_file, index=False, columns = Xtest.columns, compression='gzip')

    desc = open("../features/descriptions/"+str(num)+".txt", "w")
    desc.write("gen_polys1 = " + str(gen_polys1) +'\n')
    desc.write("stats1 = " + str(stats1) +'\n')
    desc.write("blended_preds = " + str(blended_preds) +'\n')
    desc.write("firDiff = " + str(firDiff) +'\n')
    desc.write("secDiff = " + str(secDiff) +'\n')
    desc.write("stats2 = " + str(stats2) +'\n')
    desc.write("gen_polys2 = " + str(gen_polys2) +'\n')
    desc.write("removeUseless = " + str(removeUseless) +'\n')
    desc.write("Log1p = " + str(Log) +'\n')
    desc.write("goldenFeatures = " + str(goldenFeatures) +'\n')
    desc.write("Scaling = " + str(Scaling) +'\n')
    desc.close()

def makeScatterplot():

    train_file = '../../../numerai_datasets_new/numerai_training_data.csv'
    train = pd.read_csv(train_file)


    f = train[['feature1']]
    g = train[['feature2']]
    t = train[['target']]
    ft = pd.concat([f, g, t], axis=1)

    ft_0 = ft.loc[ft['target'] == 0]
    ft_1 = ft.loc[ft['target'] == 1]

    plt.scatter(ft_1['feature1'], ft_1['feature2'])
    plt.show()

prepareDataset()
