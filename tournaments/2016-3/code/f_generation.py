import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.decomposition import PCA, FastICA
from scipy.stats import rankdata
import warnings
import dill
from qsprLib import *
#from interact_analysis import *
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sys import stdout

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
        new_col_log = np.log(preprocessing.minmax_scale(np.array(X.ix[:,i]), feature_range=[1,1000]))
        new_col_rank = preprocessing.scale(np.array(rankdata(X.ix[:,i])))
        with_added_features = np.column_stack((with_added_features, new_col_log, new_col_rank))
    reduced_38 = dim_reduce(X.ix[:, :38])
    with_added_features = np.column_stack((with_added_features, reduced_38))
    stdout.write("\n") 
    print with_added_features.shape
    print len(with_added_features[0])
    return with_added_features

def prepareDataset(seed=123, gen_polys1 = False, combinations1 = False,
        stats1 = True, createVerticalFeatures = False, blended_preds = False,
        firDiff = False, secDiff = False, 
        combinations2 = False, stats2 = False, gen_polys2 = True,
        removeUseless = False, goldenFeatures = False,
        Log = False, Scaling = False, num=19):

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
    
    if gen_polys1:
        print "Adding polynomial1 features..."
        # generate polynomial features 
        Xall = make_polynomials(Xall, Xtest=None, \
                degree=2, cutoff=100,quadratic=True)


    if combinations1:
        print "Adding combinations1..."
        features = list(Xall.columns)
        print len(features)
        for i, f in enumerate(features):
            stdout.write("\r%d out of %d" %(i,len(features)))
            stdout.flush()
            for g in features:
                if f != g:
                    if not (str(g) + "_" + str(f)) in Xall.columns:
                        Xall[str(f) + "_" + str(g)] = Xall[f] * Xall[g]
        stdout.write("\n") 

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
        #Xall['f1_mode'] = Xall.mode(axis=1)
        Xall['f1_prod'] = Xall.prod(axis=1)
        Xall['f1_skew'] = Xall.skew(axis=1)
        Xall['f1_sum'] = Xall.sum(axis=1)
        #Xall['f1_cumsum'] = Xall.cumsum(axis=1)
        #Xall['f1_cumpod'] = Xall.cumprod(axis=1)
        #Xall['f1_cummax'] = Xall.cummax(axis=1)
        #Xall['f1_cummin'] = Xall.cummin(axis=1)
        #Xall['f1_compound'] = Xall.compound(axis=1)


    if createVerticalFeatures:
        print "Adding vert features..."
        colnames = list(Xall.columns.values)
        #print colnames
        for col in colnames:
                Xall['median_'+col] = 0.0
                Xall['sdev_'+col] = 0.0
                Xall['sum_'+col] = 0.0
                Xall['count_'+col] = 0.0

        #print Xall.head()
     
    if blended_preds:
        print "Adding blended preds as features..." 
        blend_train, blend_test = f_blend.blending()
        blend_all = np.concatenate((blend_test,blend_train))
        for i in range(len(blend_all[0])):
            Xall[str(i)] = blend_all[:,i]
    
    #if gen_polys1:
    #    print "Adding polynomial1 features..."
    #    # generate polynomial features 
    #    Xall = make_polynomials(Xall, Xtest=None, \
    #            degree=2, cutoff=100,quadratic=True)

    if firDiff:
        print "Adding 1st order differential features..."
        X_diff = differentiateFeatures(Xall.iloc[:,:])
        Xall = pd.concat([Xall, X_diff],axis=1)
        Xall = Xall.ix[:,(Xall != 0).any(axis=0)]

    if secDiff:
        print "Adding 2nd order differential features..."
        X_diff = differentiateFeatures(Xall.iloc[:,:])
        Xall = pd.concat([Xall, X_diff],axis=1)
        Xall = Xall.ix[:,(Xall != 0).any(axis=0)]
    
    if combinations2:
        print "Adding combinations2..."
        features = list(Xall.columns)
        print len(features)
        for i, f in enumerate(features):
            stdout.write("\r%d out of %d" %(i,len(features)))
            stdout.flush()
            for g in features:
                if f != g:
                    if not (str(g) + "_" + str(f)) in Xall.columns:
                        Xall[str(f) + "_" + str(g)] = Xall[f] * Xall[g]
        stdout.write("\n") 

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
        Xall = make_polynomials(Xall, Xtest=None, \
                degree=2, cutoff=100,quadratic=True)


    if removeUseless:
        print "Removing low variance features..."
        # remove useless data ??
        removeLowVar(Xall, threshhold=1E-5)
    
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
        Xall = add_golden_features(Xall)
    
    if not goldenFeatures:
        print "Columns used",list(Xall.columns)
        print "Number features", len(list(Xall.columns))
    
    Xall_df = Xall 
    Xall = Xall.astype(np.float64)
    Xtrain = Xall[len(test.index):]
    Xtest = Xall[:len(test.index)]
    Xval = None
    yval = None

    if Scaling:
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
        #X[X == -inf] = 0
        #X[X == inf] = 0
        #Y[Y == -inf] = 0
        #Y[Y == inf] = 0
        with open("features__"+str(num)+".pickle", "wb") as output_file:
            dill.dump((X, training_target, Y, test_id, Xall, Xall_df), output_file)
    else:
        X = np.array(Xtrain).astype(np.float32)
        training_target = ytrain.values.T.astype(np.int32)
        Y = np.array(Xtest).astype(np.float32)
        #X[X == -inf] = 0
        #Y[Y == -inf] = 0
        with open("features__"+str(num)+".pickle", "wb") as output_file:
            dill.dump((X, training_target, Y, test_id, Xall, Xall_df), output_file)
    
    desc = open("features__"+str(num)+".txt", "w")
    desc.write("combinations1 = " + str(combinations1) +'\n')
    desc.write("stats1 = " + str(stats1) +'\n')
    desc.write("createVerticalFeatures = " + str(createVerticalFeatures) +'\n')
    desc.write("blended_preds = " + str(blended_preds) +'\n')
    desc.write("gen_polys1 = " + str(gen_polys1) +'\n')
    desc.write("firDiff = " + str(firDiff) +'\n')
    desc.write("secDiff = " + str(secDiff) +'\n')
    desc.write("combinations2 = " + str(combinations2) +'\n')
    desc.write("stats2 = " + str(stats2) +'\n')
    desc.write("gen_polys2 = " + str(gen_polys2) +'\n')
    desc.write("removeUseless = " + str(removeUseless) +'\n')
    desc.write("goldenFeatures = " + str(goldenFeatures) +'\n')
    desc.write("Scaling = " + str(Scaling) +'\n')
    desc.write("Log1p = " + str(Log) +'\n')
    desc.close()

prepareDataset()
