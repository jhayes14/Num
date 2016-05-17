#!/usr/bin/python
# coding: utf-8
from keras.initializations import one

from qsprLib import *
from lasagne_tools import *
from keras_tools import *

sys.path.append('/home/loschen/calc/smuRF/python_wrapper')
import smurf as sf

from interact_analysis import *

import matplotlib.pyplot as plt
import math
import os
pd.options.display.mpl_style = 'default'

from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.spatial.distance import euclidean, seuclidean


def prepareDataset(seed=123, nsamples=-1, holdout=False, removeZeroSales = True, specialNA = False, keepFeatures=None, dropFeatures = None, usePromoInterval = False,useStateInfo = False,imputing=False,removeTrainStoreOnly=False,oneHotenc=None,removeRare_freq=None, removeOutlier=False, other_features= False, createVerticalFeatures=False, removeClosed = False, sales_per_week=False,sales_per_day=False, logtransform = None):
    np.random.seed(seed)

    """
    types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}

    types = {'Store': np.float32, 'DayOfWeek':  np.int32, 'Sales':  np.float32,'Customers':  np.float32 , 'Promo':  np.int32, 'StateHoliday':  np.str, 'SchoolHoliday':  np.int32}
    """

    Xtrain = pd.read_csv('./data/train.csv', parse_dates=[2],low_memory=True)
    Xtrain.drop(['Customers'],axis=1,inplace=True)

    print Xtrain.describe(include='all')
    print Xtrain.shape

    Xtest = pd.read_csv('./data/test.csv', parse_dates=[3],low_memory=True)
    Xtest.loc[Xtest.Open.isnull(), 'Open'] = 1

    store = pd.read_csv('./data/store.csv')
    Xtrain = pd.merge(Xtrain, store, on='Store', how='left')
    Xtest = pd.merge(Xtest, store, on='Store', how='left')

    print "Xtrain - ISNULL:",Xtrain.isnull().any(axis=0)
    #print "Xtest - ISNULL:",Xtest.isnull().any(axis=0)
    #print "Xtrain - ISNULL:",Xtrain.Store.isnull()
    #print "Xtrain - ISNULL:",Xtrain.Store.isnull().sum()
    #raw_input()

    if removeZeroSales:
        zs = Xtrain["Sales"] > 0
        print "Removing Zero sales:",np.sum(~zs)
        Xtrain = Xtrain[zs]
    if removeClosed:
        zs = Xtrain.Open==1
        print "Removing Closed stores:",np.sum(~zs)
        Xtrain = Xtrain[zs]


    if removeOutlier:
        """
        Remove sale
        a) remove store where sales is 2times below sdev/median of specific store!
        b) remove store where sales
        """
        print "Remove Outlier!!!"
        sdev = np.std(Xtrain.Sales)
        mean = np.mean(Xtrain.Sales)
        idx = np.abs(Xtrain.Sales-mean)< 2* sdev
        print idx
        Xtrain = Xtrain[idx]




    if imputing:
        #print "Xtrain - ISNULL:",Xtrain.Store.isnull()
        #Xtrain.groupby([Xtrain.Date.dt.year,Xtrain.Date.dt.month])['Store'].count().plot(kind="bar")
        store_missing = Xtrain.groupby(['Store'])['Date'].count()<942 #length 1115
        store_missing = list(store_missing.index[store_missing.values])
        idx_missing = Xtrain['Store'].isin(store_missing)
        #print Xtrain[idx_missing]
        Xtrain['timegap'] = 0
        Xtrain.loc[idx_missing,'timegap'] = 1
        idx_missing = Xtest['Store'].isin(store_missing)

        Xtest['timegap'] = 0
        Xtest.loc[idx_missing,'timegap'] = 1

        #print Xtrain.timegap.hist()
        #plt.show()

        #g = Xtest.groupby(['Store'])['Date'].count()
        #print g

        #plt.show()
        #raw_input()

    if removeTrainStoreOnly:
        uniq_train = Xtrain['Store'].unique()
        uniq_test = Xtest['Store'].unique()
        uniq_features = compareList(uniq_train, uniq_test, verbose=False)
        if uniq_features.shape[0] > 0:
                # print Xtrain[col].value_counts()
                print "Removing unique stores values in Train/Test", uniq_features
                replace_dic = {}
                for feat in uniq_features:
                    replace_dic[feat] = 'AMBIGIOUS'
                Xtrain['Store'].replace(replace_dic, inplace=True)

    if nsamples != -1:
        if isinstance(nsamples, str) and 'shuffle' in nsamples:
            print "Shuffle train data..."
            rows = np.random.choice(len(Xtrain.index), size=len(Xtrain.index), replace=False)
        else:
            rows = np.random.choice(len(Xtrain.index), size=nsamples, replace=False)

        print "unique rows: %6.2f" % (float(np.unique(rows).shape[0]) / float(rows.shape[0]))
        Xtrain = Xtrain.iloc[rows, :]
    print Xtrain.shape

    if createVerticalFeatures:
        print "Creating Sales per Store features..."
        stores = Xtrain.groupby('Store')
        Xtrain['mean_sale_per_store'] = 0.0
        Xtest['mean_sale_per_store'] = 0.0
        Xtrain['sdev_sale_per_store'] = 0.0
        Xtest['sdev_sale_per_store'] = 0.0

        for id,indices in stores.groups.items():
            #iterate over store
            sys.stdout.write("\rStore %d"%id)
            #print Xtrain.loc[indices]
            total_mean = Xtrain.loc[indices,['Sales']].mean()
            variance = Xtrain.loc[indices,['Sales']].std()
            #print "mean:",type(total_mean)
            Xtrain.loc[indices,['mean_sale_per_store']]= total_mean.values
            Xtest.loc[Xtest.Store==id,['mean_sale_per_store']]= total_mean.values
            Xtrain.loc[indices,['sdev_sale_per_store']]= variance.values
            Xtest.loc[Xtest.Store==id,['sdev_sale_per_store']]= variance.values
            #print Xtrain.loc[indices,['mean_sale_per_store']]
            #print Xtrain.loc[indices,['Store','mean_sale_per_store']]
        #pd.DataFrame(y, columns=['discount']).to_csv('./data/discount.csv', index=False)

    if sales_per_day:
        print "\nCreating Sales per day features..."
        gr = Xtrain.groupby(['DayOfWeek','Store'])
        Xtrain['median_sale_per_day'] = 0.0
        Xtest['median_sale_per_day'] = 0.0
        Xtrain['sdev_sale_per_day'] = 0.0
        Xtest['sdev_sale_per_day'] = 0.0

        for (day_id,store_id),indices in gr.groups.items():
            sys.stdout.write("\rday %r store %r"%(day_id,store_id))
            total_mean = Xtrain.loc[indices,['Sales']].mean()
            variance = Xtrain.loc[indices,['Sales']].std()
            Xtrain.loc[indices,['median_sale_per_day']]= total_mean.values
            Xtrain.loc[indices,['sdev_sale_per_day']]= variance.values

            test_idx = np.logical_and(Xtest.DayOfWeek == day_id, Xtest.Store ==  store_id)
            Xtest.loc[test_idx,['median_sale_per_day']]= total_mean.values #get right store?
            Xtest.loc[test_idx,['sdev_sale_per_day']]= variance.values #get right store?
            #print "n stores:",(Xtest.Store == store_id).sum()
            #print Xtest.loc[test_idx,['DayOfWeek','median_sale_per_week']]

    if sales_per_week:
        print "\nCreating Sales per Week features..."
        Xtrain['WeekOfYear'] = Xtrain.Date.dt.weekofyear
        Xtest['WeekOfYear'] = Xtest.Date.dt.weekofyear
        gr = Xtrain.groupby(['WeekOfYear','Store'])
        Xtrain['median_sale_per_week'] = 0.0
        Xtest['median_sale_per_week'] = 0.0
        Xtrain['sdev_sale_per_week'] = 0.0
        Xtest['sdev_sale_per_week'] = 0.0

        for (day_id,store_id),indices in gr.groups.items():
            sys.stdout.write("\rweek %r store %r"%(day_id,store_id))
            total_mean = Xtrain.loc[indices,['Sales']].mean()
            variance = Xtrain.loc[indices,['Sales']].std()
            Xtrain.loc[indices,['median_sale_per_week']]= total_mean.values
            Xtrain.loc[indices,['sdev_sale_per_week']]= variance.values

            test_idx = np.logical_and(Xtest.WeekOfYear == day_id, Xtest.Store ==  store_id)
            Xtest.loc[test_idx,['median_sale_per_week']]= total_mean.values #get right store?
            Xtest.loc[test_idx,['sdev_sale_per_week']]= variance.values #get right store?


    #Xtrain.groupby([Xtrain.Date.dt.year,Xtrain.Date.dt.month])['Sales'].mean().plot(kind="bar")
    #Xtest.groupby([Xtest.Date.dt.year,Xtest.Date.dt.month])['Store'].mean().plot(kind="bar")

    #rearrange
    ytrain = Xtrain['Sales'].values
    Xtrain.drop(['Sales'],axis=1,inplace=True)
    test_id = Xtest['Id']
    Xtest.drop(['Id'],axis=1,inplace=True)

    #
    #Xtest.groupby(Xtrain.Date.dt.month).count().plot(kind="bar")
    #plt.show()
    #combine data
    Xall = pd.concat([Xtest, Xtrain], ignore_index=True)
    Xall['Year'] = Xall.Date.dt.year
    Xall['Month'] = Xall.Date.dt.month
    Xall['Day'] = Xall.Date.dt.day
    Xall['DayOfWeek'] = Xall.Date.dt.dayofweek
    Xall['WeekOfYear'] = Xall.Date.dt.weekofyear
    #Xall['total_months'] =  (((Xall['Year'] - 2013)* 12 +  Xall['Month']))
    Xall.drop(['Date'],axis=1,inplace=True)

    #encode
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    Xall.StoreType.replace(mappings, inplace=True)
    Xall.Assortment.replace(mappings, inplace=True)
    Xall.StateHoliday.replace(mappings, inplace=True)

    if specialNA:
        #Xall.apply(lambda x: x.fillna(x.median()),axis=0)
        Xall.fillna(Xall.median(), inplace=True)
        #pass
    else:
        Xall.fillna(0, inplace=True)

    if usePromoInterval or other_features:
        print Xall['PromoInterval']
    else:
        Xall.drop(['PromoInterval'],axis=1,inplace=True)

    if useStateInfo:
        states = pd.read_csv('./data/store_states.csv')
        Xall = pd.merge(Xall, states, on='Store', how='left')
        lbl = preprocessing.LabelEncoder()
        Xall['State'] = lbl.fit_transform(Xall['State'].values)

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




    if other_features:
        # CompetionOpen en PromoOpen from https://www.kaggle.com/ananya77041/rossmann-store-sales/randomforestpython/code
        # Calculate time competition open time in months
        Xall['CompetitionOpen'] = 12 * (Xall.Year - Xall.CompetitionOpenSinceYear) + \
            (Xall.Month - Xall.CompetitionOpenSinceMonth)
        # Promo open time in months
        Xall['PromoOpen'] = 12 * (Xall.Year - Xall.Promo2SinceYear) + \
            (Xall.WeekOfYear - Xall.Promo2SinceWeek) / 4.0
        Xall['PromoOpen'] = Xall.PromoOpen.apply(lambda x: x if x > 0 else 0)
        Xall.loc[Xall.Promo2SinceYear == 0, 'PromoOpen'] = 0

        # Indicate that sales on that day are in promo interval

        month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
                 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

        Xall['monthStr'] = Xall.Month.map(month2str)
        Xall.loc[Xall.PromoInterval == 0, 'PromoInterval'] = ''
        Xall['IsPromoMonth'] = 0
        for interval in Xall.PromoInterval.unique():
            if interval != '':
                for month in interval.split(','):
                    Xall.loc[(Xall.monthStr == month) & (Xall.PromoInterval == interval), 'IsPromoMonth'] = 1

        Xall.drop(['monthStr','PromoInterval'], axis=1, inplace=True)

    if logtransform is not None:
        print "log Transform"
        for col in logtransform:
            if col in Xall.columns: Xall[col] = Xall[col].map(np.log1p)
            # print Xall[col].describe(include=all)


    if keepFeatures is not None:
        dropcols = [col for col in Xall.columns if col not in keepFeatures]
        for col in dropcols:
            if col in Xall.columns:
                print "Dropping: ", col
                Xall.drop([col], axis=1, inplace=True)

    if dropFeatures is not None:
        for col in dropFeatures:
            if col in Xall.columns:
                print "Dropping: ", col
                Xall.drop([col], axis=1, inplace=True)

    print Xall
    print "Columns used",list(Xall.columns)



    ytrain = np.log1p(ytrain)


    #split data
    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]

    if holdout:
        print "Split holdout..."

        mask = np.logical_and(Xtrain.Year.values == 2014,Xtrain.Month.values == 8)
        mask2 = np.logical_and(Xtrain.Year.values == 2014,Xtrain.Month.values == 9)
        mask = np.logical_or(mask,mask2)
        mask3 = np.logical_and(Xtrain.Year.values == 2015,Xtrain.Month.values == 6)
        mask4 = np.logical_and(Xtrain.Year.values == 2015,Xtrain.Month.values == 7)
        mask = np.logical_or(mask,mask3)
        mask = np.logical_or(mask,mask4)
        #mask = mask4

        Xval = Xtrain[mask]
        yval = ytrain[mask]
        Xtrain = Xtrain[~mask]
        ytrain = ytrain[~mask]
        print "Shape Xtrain:",Xtrain.shape
        print "Shape Xval  :",Xval.shape

    #Xtrain.drop(['Date'],axis=1,inplace=True)
    #Xtest.drop(['Date'],axis=1,inplace=True)

    #print "Training data:",Xtrain.shape

    return Xtest, Xtrain, ytrain, test_id, None, Xval, yval


def mergeWithXval(Xtrain,Xval,ytrain,yval):
    Xtrain =  pd.concat([Xtrain, Xval], ignore_index=True)
    ytrain = np.hstack((ytrain,yval))
    return Xtrain,ytrain

class ForwardDateCV():
    def __init__(self,month, year,n_iter=5,repeats=1,useAll=True,verbose=True):
        self.year = year
        self.month = month
        self.verbose = verbose
        self.repeats = repeats
        self.n_folds = n_iter
        self.useAll = useAll

    def __len__(self):
        return self.n_folds * self.repeats

    def __iter__(self):
        month_label = (((self.year -2013)* 12 +  self.month))
        if self.useAll:
            unique_labels = np.unique(month_label)
        #unique_labels = np.asarray([27,10,5,13,22,18,12,19,9])
        else:
            unique_labels = np.asarray([7,9,10,13,18,19,22,27])# low var set!!!OK #12???

        for i in range(self.repeats):
            n = len(unique_labels)
            #s = 1.0/(float(n))
            #print "n: %d   testsize: %4.2f"%(n,s)
            cv = cross_validation.KFold(n, self.n_folds)
            #cv = cross_validation.ShuffleSplit(n=n, n_iter=self.n_folds, test_size=s)
            #print unique_labels
            #unique_labels = shuffle(unique_labels)
            #print unique_labels
            for train, test in cv:
                test_labels = unique_labels[test]
                print "test_labels:",test_labels
                test_mask = np.in1d(month_label, test_labels)
                train_mask = np.logical_not(test_mask)
                print "test sets %d train set %d"%(np.sum(test_mask),np.sum(train_mask))
                train_indices = np.where(train_mask)[0]
                test_indices = np.where(test_mask)[0]
                yield (train_indices, test_indices)



def makePredictions(model=None,Xtest=None,idx=None,filename='submission.csv',scale=0.985):
    print "Saving submission: ", filename
    if model is not None:
        log_preds = model.predict(Xtest)
    else:
        log_preds = Xtest
    if idx is None:
        idx = np.arange(Xtest.shape[0])+1

    if scale>-1:
        print "Scaling predictions: %4.2f"%(scale)
        preds = np.expm1(log_preds)*scale
    else:
        preds = np.expm1(log_preds)

    result = pd.DataFrame({"Id": idx, 'Sales': preds})
    result.to_csv(filename, index=False)


if __name__ == "__main__":
    """
    MAIN PART
    """
    plt.interactive(False)
    #TODO
    # Features
    # count features 748 942


    # CV
    # https://www.kaggle.com/c/rossmann-store-sales/forums/t/17143/inconsistent-co-relationship-between-cross-validation-and-lb
    # https://www.kaggle.com/c/rossmann-store-sales/forums/t/16862/thoughts-on-validation-testing-overfit?page=2
    # https://www.kaggle.com/c/rossmann-store-sales/forums/t/16862/thoughts-on-validation-testing-overfit

    #external data
    # https://www.kaggle.com/c/rossmann-store-sales/forums/t/17229/external-data-and-other-information?page=2

    # RMSPE
    # https://www.kaggle.com/c/rossmann-store-sales/forums/t/17601/correcting-log-sales-prediction-for-rmspe/99643#post99643
    # https://www.kaggle.com/c/rossmann-store-sales/forums/t/17649/unexpected-result

    # Other
    # https://www.kaggle.com/c/rossmann-store-sales/forums/t/17026/a-connection-between-rmspe-and-the-log-transform
    # time series: https://www.kaggle.com/c/rossmann-store-sales/forums/t/16930/anyone-started-with-time-series?page=3
    # https://www.kaggle.com/c/rossmann-store-sales/forums/t/17160/has-anyone-managed-lb-0-105-using-a-non-tree-based-method
    # https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/12947/achieve-0-50776-on-the-leaderboard-in-a-minute-with-xgboost/68472#post68472

    # feature engineering
    # putting store on the map: https://www.kaggle.com/c/rossmann-store-sales/forums/t/17048/putting-stores-on-the-map
    # https://www.kaggle.com/c/rossmann-store-sales/forums/t/17387/inputs-on-feature-engineering-for-store-prediction-regression-challenges

    # scripts
    # https://www.kaggle.com/cast42/rossmann-store-sales/xgboost-in-python-with-rmspe-v2/code
    # RMSPE=0.10361 ; https://www.kaggle.com/abhilashawasthi/rossmann-store-sales/xgb-rossmann/run/86608

    # General
    # Find CV procedure
    # identify outliers # showMisclass
    # Google trends
    # make data dummy
    # use proper data
    # use sales median for prediction
    # Create categorical features corresponding to leaf nodes ->Herra Hu
    t0 = time()

    print "numpy:", np.__version__
    print "pandas:", pd.__version__
    print "scipy:", sp.__version__

    seed = 51176
    nsamples = -1#'shuffle'
    holdout = True
    dropFeatures = None#['Store']# ['CompetitionDistance','CompetitionOpenSinceMonth']
    keepFeatures = None#['Store','DayOfWeek','Promo']
    usePromoInterval = False
    useStateInfo = False
    imputing = False
    removeTrainStoreOnly = False
    oneHotenc = ['Assortment','StoreType','DayOfWeek','CompetitionOpenSinceMonth','CompetitionOpenSinceYear']
    removeRare_freq = None
    removeOutlier = False
    specialNA = False
    other_features = False
    removeClosed = False
    createVerticalFeatures = True
    sales_per_week = False#overfit
    sales_per_day = False#overfit?
    logtransform = ['CompetitionDistance']

    Xtest, Xtrain, ytrain, idx,  sample_weight, Xval, yval = prepareDataset(seed=seed, nsamples=nsamples, holdout=holdout,keepFeatures = keepFeatures, specialNA=specialNA, dropFeatures= dropFeatures,usePromoInterval = usePromoInterval,useStateInfo=useStateInfo,imputing=imputing,removeTrainStoreOnly=removeTrainStoreOnly, oneHotenc= oneHotenc, removeRare_freq=removeRare_freq, removeOutlier= removeOutlier, other_features= other_features, createVerticalFeatures=createVerticalFeatures, removeClosed=removeClosed, sales_per_week= sales_per_week,sales_per_day=sales_per_day, logtransform=logtransform)
    if isinstance(Xtrain, pd.DataFrame):
        Xtrain.to_csv('./data/Xtrain.csv', index=False)
        Xtest.to_csv('./data/Xtest.csv', index=False)
        pd.DataFrame(ytrain, columns=['log_sales']).to_csv('./data/ytrain.csv', index=False)

    print Xtrain.head()
    #interact_analysis(Xtrain)
    #model = sf.RandomForest(n_estimators=120,mtry=5,node_size=5,max_depth=6,n_jobs=2,verbose_level=0)
    #model = Pipeline([('scaler', StandardScaler()), ('model',ross1)])
    #model = RandomForestRegressor(n_estimators=100,max_depth=None,min_samples_leaf=5,n_jobs=4, max_features=Xtrain.shape[1]/3,oob_score=False)
    #model = XgboostRegressor(n_estimators=300,learning_rate=0.3,max_depth=10, NA=0,subsample=.9,colsample_bytree=0.7,min_child_weight=5,n_jobs=1,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #model = XgboostRegressor(n_estimators=250,learning_rate=0.05,max_depth=10,subsample=.5,lambda_L2=0.001,n_jobs=1,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #model = XgboostRegressor(n_estimators=3000,learning_rate=0.02,max_depth=10, NA=-999.0,subsample=.9,colsample_bytree=0.7,min_child_weight=5,n_jobs=1,objective='reg:linear',eval_metric='rmse',booster='gbtree',silent=1,eval_size=0.0)
    #model = LinearRegression()#0.445
    #model = KNeighborsRegressor(n_neighbors=5)
    #model = ExtraTreesClassifier(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=2, max_features=3*Xtrain.shape[1]/3)

    model = KerasNNReg(dims=Xtrain.shape[1],nb_classes=1,nb_epoch=3,learning_rate=0.02,validation_split=0.0,batch_size=64,verbose=1,loss='mse')
    model = Pipeline([('scaler', StandardScaler()), ('nn',model)])

    #cv = KFold(ytrain.shape[0],8,shuffle=True)
    cv = ForwardDateCV(Xtrain.Month,Xtrain.Year,n_iter=2,repeats=1,useAll=False,verbose=True)
    #cv = StratifiedShuffleSplit(Xtrain.Year,8,shuffle=True)
    #cv = ShuffleSplit(ytrain.shape[0],8)

    scoring_func = make_scorer(root_mean_squared_percentage_error_mod, greater_is_better=False)
    #scoring_func = make_scorer(root_mean_squared_percentage_error, greater_is_better=False)
    #df = sf.DF(Xtrain.values, label=ytrain)
    #print df.printSummary()
    #parameters = {'n_estimators':[3000],'max_depth':[8,10],'learning_rate':[0.02,0.05],'subsample':[0.9],'colsample_bytree':[0.7],'min_child_weight':[5]}
    parameters = {'nn__nb_epoch':[10,20],'nn__learning_rate':[0.2 ]}
    model = makeGridSearch(model, Xtrain, ytrain, n_jobs=1, refit=True, cv=cv, scoring=scoring_func,parameters=parameters, random_iter=-1)

    #Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
    print Xtrain.shape
    model = buildModel(model,Xtrain,ytrain,cv=cv, scoring=scoring_func, n_jobs=1,trainFull=True,verbose=True)
    #model = buildXvalModel(model,Xtrain,ytrain,sample_weight=None,class_names=None,refit=True,cv=cv)

    yval_pred = model.predict(Xval)
    print "Eval-score: %5.3f"%(root_mean_squared_percentage_error(np.expm1(yval),np.expm1(yval_pred)))

    print "Training the final model (incl. Xval.)"
    Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
    #model.fit(Xtrain,ytrain)

    #makePredictions(model,Xtest,idx=idx, filename='./submissions/sub30112015.csv')

    plt.show()
    print("Model building done in %fs" % (time() - t0))
