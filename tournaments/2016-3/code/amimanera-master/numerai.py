#!/usr/bin/python
# coding: utf-8

from qsprLib import *

from interact_analysis import *
from keras_tools import *

import matplotlib.pyplot as plt
import math
import os
pd.options.display.mpl_style = 'default'


def prepareDataset(quickload=False, seed=123, nsamples=-1, holdout=False, keepFeatures=None, dropFeatures = None,dummy_encoding=None,labelEncode=None, oneHotenc=None,removeRare_freq=None, createVerticalFeatures=False, logtransform = None, polynomialFeatures=False,makeDiff=False):
    np.random.seed(seed)

    #store = pd.HDFStore('./data_numerai/store.h5')


    Xtrain = pd.read_csv('../../../../numerai_datasets/numerai_training_data.csv')

    print Xtrain.info()
    #print Xtrain.describe(include='all')
    print "Xtrain.shape:",Xtrain.shape

    Xtest = pd.read_csv('../../../../numerai_datasets/numerai_tournament_data.csv')
    test_id = Xtest['t_id']
    Xtest.drop(['t_id'],axis=1,inplace=True)
    Xtest['validation']=-1
    print "Xtest.shape:",Xtest.shape


    if nsamples != -1:
        if isinstance(nsamples, str) and 'shuffle' in nsamples:
            print "Shuffle train data..."
            rows = np.random.choice(len(Xtrain.index), size=len(Xtrain.index), replace=False)
        else:
            rows = np.random.choice(len(Xtrain.index), size=nsamples, replace=False)

        print "unique rows: %6.2f" % (float(np.unique(rows).shape[0]) / float(rows.shape[0]))
        Xtrain = Xtrain.iloc[rows, :]
    print Xtrain.shape



    ytrain = Xtrain['target']
    Xtrain.drop(['target'],axis=1,inplace=True)

    print "Xtrain - ISNULL:",Xtrain.isnull().any(axis=0)
    print "Xtest - ISNULL:",Xtest.isnull().any(axis=0)


    Xall = pd.concat([Xtest, Xtrain], ignore_index=True)


    if dummy_encoding is not None:
        print "Dummy encoding,skip label encoding"
        Xall = pd.get_dummies(Xall,columns=dummy_encoding)

    if labelEncode is not None:
        print "Label encode"
        for col in labelEncode:
            lbl = preprocessing.LabelEncoder()
            Xall[col] = lbl.fit_transform(Xall[col].values)
            vals = Xall[col].unique()
            print "Col: %s Vals %r:"%(col,vals)
            print "Orig:",list(lbl.inverse_transform(Xall[col].unique()))

    if removeRare_freq is not None:
        print "Remove rare features based on frequency..."
        for col in oneHotenc:
            ser = Xall[col]
            counts = ser.value_counts().keys()
            idx = ser.value_counts() > removeRare_freq
            threshold = idx.astype(int).sum()
            print "%s has %d different values before, min freq: %d - threshold %d" % (
                col, len(counts), removeRare_freq, threshold)
            if len(counts) > threshold:
                ser[~ser.isin(counts[:threshold])] = 9999
            if len(counts) <= 1:
                print("Dropping Column %s with %d values" % (col, len(counts)))
                Xall = Xall.drop(col, axis=1)
            else:
                Xall[col] = ser.astype('category')
            print ser.value_counts()
            counts = ser.value_counts().keys()
            print "%s has %d different values after" % (col, len(counts))


    if createVerticalFeatures:
        print "Creating vert features..."
        #colnames = ['f1', 'f2', 'f3']
        colnames = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14']
        print Xall.columns
        gr = Xall.groupby('c1')
        for col in colnames:
                Xall['median_'+col] = 0.0
                Xall['sdev_'+col] = 0.0
                Xall['sum_'+col] = 0.0
                Xall['count_'+col] = 0.0

        for id,indices in gr.groups.items():
            for col in colnames:
                total_mean = Xall.loc[indices,[col]].median()
                variance = Xall.loc[indices,[col]].std()
                count = Xall.loc[indices,[col]].shape[0]
                sum = Xall.loc[indices,[col]].sum()
                Xall.loc[indices,['median_'+col]]= total_mean.values
                Xall.loc[indices,['sdev_'+col]]= variance.values
                Xall.loc[indices,['count_'+col]]= count
                Xall.loc[indices,['sum_'+col]]= sum.values


        print Xall.head()

    if polynomialFeatures:
        print "Polynomial feature of degree:", polynomialFeatures
        if isinstance(polynomialFeatures, str) and 'load' in polynomialFeatures:
            X_poly = pd.read_csv('poly.csv').reset_index(drop=True)
            print X_poly.describe()
        else:
            X_poly = make_polynomials(Xall[polynomialFeatures],degree=2,cutoff=100,quadratic=True)
            X_poly.to_csv('poly.csv')

        print X_poly.head()
        Xall = pd.concat([Xall, X_poly], axis=1)


    if oneHotenc is not None:
        print "1-0 Encoding categoricals...", oneHotenc
        for col in oneHotenc:
            #print "Unique values for col:", col, " -", np.unique(Xall[col].values)
            encoder = OneHotEncoder()
            X_onehot = pd.DataFrame(encoder.fit_transform(Xall[[col]].values).todense())
            X_onehot.columns = [col + "_" + str(column) for column in X_onehot.columns]
            print "One-hot-encoding of %r...new shape: %r" % (col, X_onehot.shape)
            Xall.drop([col], axis=1, inplace=True)
            Xall = pd.concat([Xall, X_onehot], axis=1)
            print "One-hot-encoding final shape:", Xall.shape
            # raw_input()

    if makeDiff is not None:
        X_diff = differentiateFeatures(Xall.iloc[:,:14])
        if '2nd' in makeDiff:
            X_diff = differentiateFeatures(X_diff)
        Xall = pd.concat([Xall, X_diff],axis=1)
        Xall = Xall.ix[:,(Xall != 0).any(axis=0)]

    if logtransform is not None:
        print "log Transform"
        for col in logtransform:
            if col in Xall.columns: Xall[col] = Xall[col].map(np.log1p)
            # print Xall[col].describe(include=all)



    if keepFeatures is not None:
        dropcols = [col for col in Xall.columns if col not in keepFeatures]
        for col in dropcols:
            if col in Xall.columns and not 'validation' in col:
                print "Dropping: ", col
                Xall.drop([col], axis=1, inplace=True)

    if dropFeatures is not None:
        for col in dropFeatures:
            if col in Xall.columns:
                print "Dropping: ", col
                Xall.drop([col], axis=1, inplace=True)



    print "Columns used",list(Xall.columns)


    #split data
    Xall = Xall.astype(np.float64)
    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]
    Xval = None
    yval = None

    if holdout:
        print "Split holdout..."
        print Xtrain.shape
        #print Xval.shape
        print ytrain.shape
        #Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size=0.25, random_state=42)
        train_mask = (Xtrain.validation == 0).values
        val_mask = (Xtrain.validation == 1).values


        Xval = Xtrain.loc[val_mask,:]
        yval = ytrain[val_mask]

        Xtrain = Xtrain.loc[train_mask,:]
        ytrain = ytrain[train_mask]

        print "Shape Xtrain:",Xtrain.shape
        print "Shape Xval  :",Xval.shape

    Xtrain.drop(['validation'],axis=1,inplace=True)
    Xval.drop(['validation'],axis=1,inplace=True)
    Xtest.drop(['validation'],axis=1,inplace=True)

    print "Training data:",Xtrain.info()

    #store['Xtest'] = Xtest
    #store['Xtrain'] = Xtrain
    #store['ytrain'] = ytrain
    #store['Xval'] = Xval
    #store['yval'] = yval
    #store['test_id'] = test_id
    #print store
    #store.close()



    return Xtest, Xtrain, ytrain.values, test_id, None, Xval, yval.values


def mergeWithXval(Xtrain,Xval,ytrain,yval):
    Xtrain =  pd.concat([Xtrain, Xval], ignore_index=True)
    ytrain = np.hstack((ytrain,yval))
    return Xtrain,ytrain


def makePredictions(model=None,Xtest=None,idx=None,filename='submission.csv'):
    print "Saving submission: ", filename
    if model is not None:
        preds = model.predict_proba(Xtest)[:,1]
    else:
        preds = Xtest
    if idx is None:
        idx = np.arange(Xtest.shape[0])+1

    result = pd.DataFrame({"t_id": idx, 'probability': preds})
    result.to_csv(filename, index=False)


if __name__ == "__main__":
    """
    MAIN PART
    """
    plt.interactive(False)
    #TODO

    t0 = time()

    print "numpy:", np.__version__
    print "pandas:", pd.__version__
    print "scipy:", sp.__version__

    numericals = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14']
    interactions = ['f1xf2', 'f1xf3', 'f1xf4', 'f1xf5', 'f1xf6', 'f1xf7', 'f1xf8', 'f1xf9', 'f1xf10', 'f1xf11', 'f1xf12', 'f1xf13', 'f1xf14', 'f2xf3', 'f2xf4', 'f2xf5', 'f2xf6', 'f2xf7', 'f2xf8', 'f2xf9', 'f2xf10', 'f2xf11', 'f2xf12', 'f2xf13', 'f2xf14', 'f3xf4', 'f3xf5', 'f3xf6', 'f3xf7', 'f3xf8', 'f3xf9', 'f3xf10', 'f3xf11', 'f3xf12', 'f3xf13', 'f3xf14', 'f4xf5', 'f4xf6', 'f4xf7', 'f4xf8', 'f4xf9', 'f4xf10', 'f4xf11', 'f4xf12', 'f4xf13', 'f4xf14', 'f5xf6', 'f5xf7', 'f5xf8', 'f5xf9', 'f5xf10', 'f5xf11', 'f5xf12', 'f5xf13', 'f5xf14', 'f6xf7', 'f6xf8', 'f6xf9', 'f6xf10', 'f6xf11', 'f6xf12', 'f6xf13', 'f6xf14', 'f7xf8', 'f7xf9', 'f7xf10', 'f7xf11', 'f7xf12', 'f7xf13', 'f7xf14', 'f8xf9', 'f8xf10', 'f8xf11', 'f8xf12', 'f8xf13', 'f8xf14', 'f9xf10', 'f9xf11', 'f9xf12', 'f9xf13', 'f9xf14', 'f10xf11', 'f10xf12', 'f10xf13', 'f10xf14', 'f11xf12', 'f11xf13', 'f11xf14', 'f12xf13', 'f12xf14', 'f13xf14']
    categories=['c1_0', 'c1_1', 'c1_2', 'c1_3', 'c1_4', 'c1_5', 'c1_6', 'c1_7', 'c1_8', 'c1_9', 'c1_10', 'c1_11', 'c1_12', 'c1_13', 'c1_14', 'c1_15', 'c1_16', 'c1_17', 'c1_18', 'c1_19', 'c1_20', 'c1_21', 'c1_22']
    verticals=['median_f1', 'sdev_f1', 'sum_f1', 'count_f1', 'median_f2', 'sdev_f2', 'sum_f2', 'count_f2', 'median_f3', 'sdev_f3', 'sum_f3', 'count_f3', 'median_f4', 'sdev_f4', 'sum_f4', 'count_f4', 'median_f5', 'sdev_f5', 'sum_f5', 'count_f5', 'median_f6', 'sdev_f6', 'sum_f6', 'count_f6', 'median_f7', 'sdev_f7', 'sum_f7', 'count_f7', 'median_f8', 'sdev_f8', 'sum_f8', 'count_f8', 'median_f9', 'sdev_f9', 'sum_f9', 'count_f9', 'median_f10', 'sdev_f10', 'sum_f10', 'count_f10', 'median_f11', 'sdev_f11', 'sum_f11', 'count_f11', 'median_f12', 'sdev_f12', 'sum_f12', 'count_f12', 'median_f13', 'sdev_f13', 'sum_f13', 'count_f13', 'median_f14', 'sdev_f14', 'sum_f14', 'count_f14']
    greedy2 = ['f9', 'f6', 'c1_8', 'c1_14', 'c1_2', 'c1_12', 'f7', 'c1_5', 'c1_6', 'c1_1', 'f1', 'c1_21', 'f2xf10', 'f11', 'f11xf13']#via linear! regression AUC=0.5356
    greedy1 = ['f1', 'f6', 'c1_c1_10', 'f3', 'f5', 'c1_c1_12', 'c1_c1_3', 'c1_c1_16', 'c1_c1_5', 'c1_c1_11', 'f11', 'f7', 'f4', 'c1_c1_7', 'c1_c1_9', 'c1_c1_20', 'f13', 'c1_c1_24', 'f14']
    greedy3 = ['f9', 'f6', 'c1_8', 'c1_14', 'c1_2', 'c1_12', 'f7', 'c1_5', 'c1_6', 'c1_1', 'f1', 'c1_21', 'count_f4', 'c1_17', 'count_f6']#via logistic regression AUC=0.5333
    greedy4 = ['count_f5', 'f1', 'f11', 'f3', 'f6', 'sdev_f9', 'f14', 'sdev_f10', 'count_f4']#via XGboost
    greedy5 = ['count_f5', 'f1', 'f11', 'f3', 'f6', 'sdev_f9', 'f14', 'sdev_f10', 'count_f4', 'f7xf13', 'f1xf4']#via XGboost AUC=0.5363

    quickload = False
    seed = 51176
    nsamples = 'shuffle'
    holdout = True
    makeDiff = '1st'
    dummy_encoding = None#['c1']
    labelEncode = ['c1']
    oneHotenc = ['c1']
    removeRare_freq = None
    createVerticalFeatures = False#True
    logtransform = numericals
    polynomialFeatures = numericals#numericals
    keepFeatures = None#greedy5
    dropFeatures = None# ['c1_0', 'c1_1', 'c1_2', 'c1_3', 'c1_4', 'c1_5', 'c1_6', 'c1_7', 'c1_8', 'c1_9', 'c1_10', 'c1_11', 'c1_12', 'c1_13', 'c1_14', 'c1_15', 'c1_16', 'c1_17', 'c1_18', 'c1_19', 'c1_20', 'c1_21', 'c1_22']
    Xtest, Xtrain, ytrain, idx,  sample_weight, Xval, yval = prepareDataset(quickload=quickload,seed=seed, nsamples=nsamples, holdout=holdout,keepFeatures = keepFeatures, dropFeatures= dropFeatures, dummy_encoding=dummy_encoding, labelEncode=labelEncode, oneHotenc= oneHotenc, removeRare_freq=removeRare_freq,  logtransform=logtransform, createVerticalFeatures=createVerticalFeatures,polynomialFeatures=polynomialFeatures,makeDiff=makeDiff)
    print list(Xtrain.columns)
    #Xtrain.iloc[:10,:-1].plot(kind='area',stacked=False)
    #Xtrain.hist(bins=100)
    plt.show()
    #interact_analysis(Xtrain)

    #model = sf.RandomForest(n_estimators=120,mtry=5,node_size=5,max_depth=6,n_jobs=2,verbose_level=0)
    #model = Pipeline([('scaler', StandardScaler()), ('model',ross1)])
    model = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=5,n_jobs=2, max_features=Xtrain.shape[1]/3,oob_score=False)
    #model = XgboostClassifier(n_estimators=100,learning_rate=0.01,max_depth=3, NA=0,subsample=.75,colsample_bytree=1.0,min_child_weight=5,n_jobs=2,objective='binary:logistic',eval_metric='logloss',booster='gbtree',silent=1,eval_size=0.0)
    #model = XgboostRegressor(n_estimators=300,learning_rate=0.3,max_depth=10, NA=0,subsample=.9,colsample_bytree=1.0,min_child_weight=5,n_jobs=1,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #model = ExtraTreesClassifier(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=2, max_features=3*Xtrain.shape[1]/3)
    model = KNeighborsClassifier(n_neighbors=20)
    #model = LinearRegression()
    #model = Pipeline([('scaler', StandardScaler()),('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', LinearRegression())])
    #model = Pipeline([('scaler', StandardScaler()),('filter', GenericUnivariateSelect(f_regression, param=95,mode='percentile')), ('model', SVC(C=1))])
    #model = Pipeline([('pca', PCA(n_components=10)),('model', Ridge())])
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')),('model', ElasticNet(alpha=.01,l1_ratio=0.001,max_iter=1000)) ])
    #model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', PLSRegression(n_components=20))])
    #model = RidgeClassifier()
    #model = Ridge()
    #model = LogisticRegression(C=10,penalty='l2')
    #model = LogisticRegression(C=100,penalty='l1')
    #model = LinearSVC()
    #model = GaussianNB()
    #model = KerasNN(dims=Xtrain.shape[1]*0.9,nb_classes=2,nb_epoch=3,validation_split=0.0,batch_size=64,verbose=1,loss='categorical_crossentropy')
    #model = BaggingClassifier(base_estimator=model,n_estimators=2,n_jobs=1,verbose=2,random_state=None,max_samples=0.9,max_features=0.9,bootstrap=True)
    model = Pipeline([('scaler', StandardScaler()), ('m',model)])
    #cv = StratifiedKFold(ytrain,8,shuffle=True)
    # cv = KFold(X.shape[0], n_folds=folds,shuffle=True)
    cv = StratifiedShuffleSplit(ytrain,n_iter=8,test_size=0.1)

    #parameters = {'n_estimators':[100],'max_depth':[3],'learning_rate':[0.01],'subsample':[0.5],'colsample_bytree':[1.0],'min_child_weight':[10,20,50,100]}
    #parameters = {'n_estimators':[150],'min_samples_leaf':[5,10,15],'max_features':[100],'criterion':['entropy']}
    #parameters = {'m__nb_epoch':[10,20,30,40],'m__learning_rate':[0.2,0.02]}
    parameters = {'m__n_neighbors':[5,10,15]}
    #parameters = {'filter__param':[99,95]}
    #parameters = {'m__C':[10],'m__penalty':['l2']}
    #parameters = {'m__C':[100],'m__gamma':[1.0/Xtrain.shape[1]]}
    model = makeGridSearch(model, Xtrain, ytrain, n_jobs=4, refit=True, cv=cv, scoring='roc_auc',parameters=parameters, random_iter=-1)
    print model
    #greedyFeatureSelection(model, Xtrain, ytrain, itermax=30, itermin=20, pool_features=None, start_features=greedy4,verbose=True, cv=cv, n_jobs=4, scoring_func='roc_auc')

    #Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
    print Xtrain.shape
    #model = buildModel(model,Xtrain,ytrain,cv=StratifiedKFold(ytrain,8,shuffle=True), scoring='roc_auc', n_jobs=8,trainFull=True,verbose=True)
    #analyzeLearningCurve(model, Xtrain, ytrain, cv=cv, score_func='roc_auc')
    #model = buildXvalModel(model,Xtrain,ytrain,sample_weight=None,class_names=None,refit=True,cv=cv)

    model.fit(Xtrain,ytrain)
    yval_pred = model.predict_proba(Xval)[:,1]
    print yval_pred
    print "Eval-score: %5.3f"%(roc_auc_score(yval,yval_pred))

    print "Training the final model (incl. Xval.)"
    Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
    model.fit(Xtrain,ytrain)

    makePredictions(model,Xtest,idx=idx, filename='./submissions/numJan16a.csv')

    plt.show()
    print("Model building done in %fs" % (time() - t0))
