#!/usr/bin/python
# coding: utf-8

from qsprLib import *
from lasagne_tools import *
from keras_tools import *

sys.path.append('/home/loschen/calc/smuRF/python_wrapper')
import smurf as sf

sys.path.append('/home/loschen/programs/xgboost/wrapper')
import xgboost as xgb

from interact_analysis import *

import matplotlib.pyplot as plt
import math
import os
pd.options.display.mpl_style = 'default'


def patient_activity():
    df_act = pd.read_csv('./data/patient_activity_head.csv',engine='c',dtype={'patient_id': np.uint32, 'activity_type': np.dtype("a1"), 'activity_year':np.uint16, 'activity_month': np.uint8})
    df_act.activity_type.replace({'R':0, 'A':1}, inplace=True)
    df_act.activity_type = df_act.activity_type.astype(np.bool)
    df_act.info()
    large = pd.HDFStore('./data/store_large.h5')
    large['df_act']=df_act
    large.close()

def create_df():
    steps = 20
    types = {'patient_id': np.dtype(np.int),
         'diagnosis_code': np.dtype(str)}
    df_chunks = pd.read_csv("/home/loschen/Desktop/datamining-kaggle/genentech/data/diagnosis_head.csv",chunksize=10**6, dtype = types)
    store = pd.HDFStore('./data/diagnosis.h5')
    label = 'diagnosis_code_0'
    for i,chunk in enumerate(df_chunks):
        print i
        #print chunk.info()
        #print chunk.head()
        chunk = chunk[['patient_id','diagnosis_code']]
        chunk = chunk.reindex(copy = False)

        if i%steps==0:
            label = 'diagnosis_code_'+str(i)
            print store
        store.append(label,chunk,index = False)
        #if i>6: break

def merge_diagnosis(X=None,iterations=2,removeRare_freq = 1000, max_features = 100, loadit=False, diagnosisOnly = False):
    store = pd.HDFStore('./data/diagnosis.h5')
    print store
    if X is not None and diagnosisOnly:
        col_list = X.columns
        col_list = col_list.remove('patient_id')
        X.drop(col_list,axis=1,inplace=True)
    #labels = ['diagnosis_code_a','diagnosis_code_b','diagnosis_code_c']
    #labels = ['diagnosis_code_a']
    if not loadit:
        for i,col in enumerate(store.keys()):
            if i>iterations: break
            if col.startswith("/X"): continue
            df = store[col]
            #label encode diagnoses
            df = removeLowFreq(df, ['diagnosis_code'], removeRare_freq = removeRare_freq, discard_rows= False, fillNA = 'rare')
            #http://stackoverflow.com/questions/27298178/concatenate-strings-from-several-rows-using-pandas-groupby
            print "Iteration: %d Grouping by... "%(i),
            df = pd.DataFrame(df.groupby('patient_id',sort=False)['diagnosis_code'].apply(lambda x: ' '.join(x)))
            #print df.head(20)
            #print df.info()

            if X is not None:
                print "...Merging..."
                X = pd.merge(X,df,left_on='patient_id',how='left',right_index=True)
                #join diagnosis code table
                if 'diagnosis_code_y' in X.columns:
                    X.rename(columns={'diagnosis_code_x':'diagnosis_code'},inplace=True)
                    X['diagnosis_code'] = X['diagnosis_code'].map(str) + " " + X['diagnosis_code_y'].astype(str)
                    #X['diagnosis_code'] = X[['diagnosis_code', 'diagnosis_code_y']].apply(lambda x: ' '.join(x), axis=1)
                    X.drop(['diagnosis_code_y'],axis=1,inplace=True)

            #print X[['patient_id','diagnosis_code']].head(20)
            #print X[['patient_id','diagnosis_code']].tail(20)
        #save x here...
        #store['X'] = X
        X.to_csv("./data/X_diagnosis.csv", index=False)
    else:
        X = pd.read_csv("./data/X_diagnosis.csv")
        print X.info()


    vectorizer = CountVectorizer(min_df=10, max_features=max_features, lowercase=True, analyzer="word",
                                         ngram_range=(1, 1), stop_words=None, strip_accents='unicode',
                                         token_pattern=r'\S{2,}\s{1,1}',dtype=np.int32)
    diag = X['diagnosis_code']
    X.drop(['diagnosis_code'],axis=1,inplace=True)
    Xtmp = vectorizer.fit_transform(diag).todense()#.astype(np.int32)
    column_names = vectorizer.get_feature_names()
    Xtmp = pd.DataFrame(Xtmp, columns=column_names)
    keys = vectorizer.vocabulary_.keys()
    print "Length of dic:", len(keys)
    print "New features:", Xtmp.shape
    print "Columns:", column_names
    X = pd.concat([X, Xtmp], axis=1)
    #print X[column_names].describe()
    return X


def prepareDataset(quickload=False, seed=123, nsamples=-1, holdout=False, keepFeatures=None, dropFeatures = None,dummy_encoding=None,labelEncode=None, oneHotenc=None,removeRare_freq=None, createVerticalFeatures=False, logtransform = None, exludePatients=True, useActivity=False, diagnosis = False):
    np.random.seed(seed)

    large = pd.HDFStore('./data/store_large.h5')

    print large

    if isinstance(quickload,str):
        store = pd.HDFStore(quickload)
        print store
        Xtest = store['Xtest']
        Xtrain = store['Xtrain']
        ytrain = store['ytrain']
        Xval = store['Xval']
        yval = store['yval']
        test_id = store['test_id']

        return Xtest, Xtrain, ytrain.values, test_id, None, Xval, yval.values

    store = pd.HDFStore('./data/store.h5')
    print store

    Xtrain = pd.read_csv('./data/patients_train.csv')
    print Xtrain.info()
    Xtrain.drop(['patient_gender'],axis=1,inplace=True)


    print Xtrain.describe(include='all')
    print "Xtrain.shape:",Xtrain.shape

    Xtest = pd.read_csv('./data/patients_test.csv')
    Xtest.drop(['patient_gender'],axis=1,inplace=True)
    print "Xtest.shape:",Xtest.shape

    #exclude
    if exludePatients:
        print "Excluding patients:"
        do_not_use_train = pd.read_csv('./data/train_patients_to_exclude.csv',header=None,index_col=False,squeeze=True)
        Xtrain = Xtrain.loc[~(Xtrain.patient_id.isin(list(do_not_use_train.values))),:]
        #do_not_use_test = pd.read_csv('./data/test_patients_to_exclude.csv',header=None,index_col=False,squeeze=True)
        #Xtest = Xtest.loc[~(Xtest.patient_id.isin(do_not_use_test.values)),:]
        print "Xtrain.shape:",Xtrain.shape
        print "Xtest.shape:",Xtest.shape

    print "Xtrain - ISNULL:",Xtrain.isnull().any(axis=0)
    print "Xtest - ISNULL:",Xtest.isnull().any(axis=0)


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
        pass


    #Xtrain.groupby([Xtrain.Date.dt.year,Xtrain.Date.dt.month])['Sales'].mean().plot(kind="bar")
    #Xtest.groupby([Xtest.Date.dt.year,Xtest.Date.dt.month])['Store'].mean().plot(kind="bar")

    #rearrange
    ytrain = Xtrain['is_screener']
    Xtrain.drop(['is_screener'],axis=1,inplace=True)
    test_id = Xtest['patient_id']

    Xall = pd.concat([Xtest, Xtrain], ignore_index=True)


    if diagnosis:
        Xall = merge_diagnosis(Xall,iterations=35,removeRare_freq = 500, max_features = 200, loadit = True)
        # 2 ~ 0.74
        # 10 ~ 0.812
        # 20 ~ 0.846

    if useActivity:
        print "Reading patient activity..."
        df_count = large['df_count']
        df_count1 = df_count.loc[df_count.activity_type==0,:]
        Xall = pd.merge(Xall, df_count1[['patient_id','activity_year']], on='patient_id', how='left')
        df_count2 = df_count.loc[df_count.activity_type==1,:]
        Xall = pd.merge(Xall, df_count2[['patient_id','activity_year']], on='patient_id', how='left')
        Xall.fillna(0, inplace=True)
        print Xall.head()
        print Xall.info()

        """
        (262158389, 4)
            patient_id activity_type  activity_year  activity_month
        0   103121024             A           2008               5
        1   209481527             R           2009              11
        2   209482911             R           2013               2
        3   209484601             A           2012               2
        4   209485106             A           2012               3
        """

    Xall.drop(['patient_id'],axis=1,inplace=True)

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

    #Xall = Xall.astype(np.float32)
    print "Columns used",list(Xall.columns)


    #split data
    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]
    Xval = None
    yval = None

    if holdout:
        print "Split holdout..."
        print Xtrain.shape
        #print Xval.shape
        print ytrain.shape
        Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size=0.25, random_state=42)

        print "Shape Xtrain:",Xtrain.shape
        print "Shape Xval  :",Xval.shape


    print "Training data:",Xtrain.info()

    df_list = [Xtest,Xtrain,ytrain,Xval,yval,test_id]
    name_list = ['Xtest','Xtrain','ytrain','Xval','yval','test_id']
    for label,ldf in zip(name_list,df_list):
        print "Store:",label
        ldf = ldf.reindex(copy = False)
        store.put(label, ldf, format='table', data_columns=True)

    print store
    store.close()

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

    result = pd.DataFrame({"patient_id": idx, 'predict_screener': preds})
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

    best30 = ['v72.31 ', 'v70.0 ', 'patient_age_group', 'rare ', 'nan ', 'v22.1 ', 'v76.12 ', 'X616.10 ', 'X401.9 ', 'v27.0 ', 'X272.4 ', 'v58.69 ', 'X780.79 ', 'X244.9 ', 'household_income', 'X789.00 ', 'X729.5 ', 'X599.0 ', 'education_level', 'X724.2 ', 'X626.2 ', 'v04.81 ', 'X530.81 ', 'X401.1 ', 'X311 ', 'X786.50 ', 'X784.0 ', 'X719.46 ', 'X272.0 ', 'X625.9 ', 'X786.2 ', 'X611.72 ', 'v76.51 ', 'X723.1 ', 'X300.00 ', 'X305.1 ', 'X250.00 ', 'X465.9 ', 'X719.41 ', 'X462 ', 'X461.9 ', 'X733.90 ', 'X268.9 ', 'X285.9 ', 'v22.2 ', 'X466.0 ', 'X493.90 ', 'X477.9 ', 'X724.5 ', 'X278.00 ', 'X780.4 ', 'X719.45 ', 'X786.05 ', 'X787.91 ', 'X785.1 ', 'X729.1 ', 'X620.2 ', 'X789.09 ', 'X788.1 ', 'X719.47 ', 'v57.1 ', 'X272.2 ', 'X346.90 ', 'patient_state_9', 'X787.01 ', 'X780.52 ', 'X782.0 ', 'X473.9 ', 'X789.06 ', 'X786.59 ', 'X564.00 ', 'X786.09 ', 'X278.01 ', 'X174.9 ', 'X724.4 ', 'X327.23 ', 'X722.52 ', 'X787.02 ', 'X782.3 ', 'patient_state_20', 'patient_state_10', 'X280.9 ', 'X723.4 ', 'X722.10 ', 'X486 ', 'X250.02 ', 'X496 ', 'X715.96 ', 'X477.0 ', 'X721.3 ', 'patient_state_37', 'patient_state_38', 'X780.2 ', 'patient_state_34', 'patient_state_35', 'X728.85 ', 'ethinicity_2', 'v58.83 ', 'patient_state_47', 'X338.29 ', 'patient_state_4', 'X739.2 ', 'X714.0 ', 'X477.8 ', 'X739.3 ', 'X414.01 ', 'ethinicity_1', 'ethinicity_0', 'patient_state_31', 'X739.1 ', 'v58.61 ', 'patient_state_43', 'X427.31 ', 'patient_state_18', 'patient_state_5', 'patient_state_6', 'ethinicity_3', 'patient_state_36', 'patient_state_22', 'patient_state_27', 'patient_state_1', 'patient_state_45', 'patient_state_3', 'X428.0 ', 'patient_state_23', 'patient_state_15', 'patient_state_14', 'patient_state_19', 'patient_state_8', 'patient_state_40', 'patient_state_42', 'patient_state_24', 'patient_state_17', 'patient_state_44', 'patient_state_12', 'patient_state_49', 'patient_state_26', 'patient_state_48', 'patient_state_25', 'patient_state_2', 'patient_state_29', 'patient_state_33', 'patient_state_16', 'patient_state_21', 'patient_state_7', 'patient_state_0', 'X285.21 ', 'X585.6 ', 'patient_state_11', 'X588.81 ', 'patient_state_32', 'patient_state_39', 'patient_state_13', 'patient_state_30', 'patient_state_46', 'patient_state_50', 'patient_state_41', 'patient_state_28']


    #new = []
    #for x in best30:
    #    if x.endswith('.'):
    #        x = list(x)
    #        x[-1] = ' '
    #        x = "".join(x)
    #    new.append(x)
    #best30 = new
    #print new
    #raw_input()
    quickload = False# './data/store_db3.h5'
    seed = 51176
    nsamples = -1#'shuffle'
    holdout = True
    dropFeatures = None
    keepFeatures = None#best30
    dummy_encoding = None#['patient_age_group','patient_state','ethinicity','household_income','education_level']
    labelEncode = ['patient_age_group','patient_state','ethinicity','household_income','education_level']
    oneHotenc = ['patient_state','ethinicity']#None#['patient_age_group','patient_state','ethinicity','household_income','education_level']
    removeRare_freq = None
    createVerticalFeatures = False
    logtransform = None
    useActivity = True
    diagnosis = True

    Xtest, Xtrain, ytrain, idx,  sample_weight, Xval, yval = prepareDataset(quickload=quickload,seed=seed, nsamples=nsamples, holdout=holdout,keepFeatures = keepFeatures, dropFeatures= dropFeatures, dummy_encoding=dummy_encoding, labelEncode=labelEncode, oneHotenc= oneHotenc, removeRare_freq=removeRare_freq,  logtransform=logtransform, useActivity= useActivity, diagnosis = diagnosis)

    #interact_analysis(Xtrain)
    #model = sf.RandomForest(n_estimators=120,mtry=5,node_size=5,max_depth=6,n_jobs=2,verbose_level=0)
    #model = Pipeline([('scaler', StandardScaler()), ('model',ross1)])
    #model = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=5,n_jobs=2, max_features=Xtrain.shape[1]/3,oob_score=False)
    #model = XgboostClassifier(n_estimators=800,learning_rate=0.025,max_depth=10, NA=0,subsample=.9,colsample_bytree=0.7,min_child_weight=5,n_jobs=2,objective='binary:logistic',eval_metric='logloss',booster='gbtree',silent=1,eval_size=0.0)
    model = XgboostClassifier(n_estimators=200,learning_rate=0.2,max_depth=10, NA=0,subsample=.9,colsample_bytree=0.7,min_child_weight=5,n_jobs=2,objective='binary:logistic',eval_metric='logloss',booster='gbtree',silent=1,eval_size=0.0)
    #model = ExtraTreesClassifier(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=2, max_features=3*Xtrain.shape[1]/3)
    #model = KNeighborsClassifier(n_neighbors=20)
    #model = LogisticRegression()

    #model = KerasNN(dims=Xtrain.shape[1],nb_classes=2,nb_epoch=3,learning_rate=0.02,validation_split=0.0,batch_size=128,verbose=1,layers=[32,32], dropout=[0.2,0.2])
    #model = Pipeline([('scaler', StandardScaler()), ('nn',model)])
    cv = StratifiedKFold(ytrain,2,shuffle=True)
    # cv =
    # cv = KFold(X.shape[0], n_folds=folds,shuffle=True)
    #cv = StratifiedShuffleSplit(ytrain,2)
    #scoring_func = roc_auc_score
    #print df.printSummary()
    parameters = {'n_estimators':[800],'max_depth':[10,20],'learning_rate':[0.025,0.05,0.01],'subsample':[0.7,0.9],'colsample_bytree':[0.7,0.9],'min_child_weight':[5]}
    #parameters = {'nn__nb_epoch':[10,20],'nn__learning_rate':[0.2 ]}
    #parameter={}
    #model = makeGridSearch(model, Xtrain, ytrain, n_jobs=2, refit=True, cv=cv, scoring='roc_auc',parameters=parameters, random_iter=-1)

    #Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
    print Xtrain.shape
    model = buildModel(model,Xtrain,ytrain,cv=cv, scoring='roc_auc', n_jobs=2,trainFull=True,verbose=True)
    #analyzeLearningCurve(model, Xtrain, ytrain, cv=cv, score_func='roc_auc')
    #model = buildXvalModel(model,Xtrain,ytrain,sample_weight=NoneFalse,class_names=None,refit=True,cv=cv)

    print "Evaluation data set..."
    model.fit(Xtrain,ytrain)
    yval_pred = model.predict_proba(Xval)[:,1]
    print "Eval-score: %5.3f"%(roc_auc_score(yval,yval_pred))

    print "Training the final model (incl. Xval.)"
    Xtrain, ytrain = mergeWithXval(Xtrain,Xval,ytrain,yval)
    model.fit(Xtrain,ytrain)

    makePredictions(model,Xtest,idx=idx, filename='./submissions/gensubu25012016.csv')


    plt.show()
    print("Model building done in %fs" % (time() - t0))
