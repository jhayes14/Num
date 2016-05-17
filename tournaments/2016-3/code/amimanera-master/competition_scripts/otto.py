#!/usr/bin/python 
# coding: utf-8

"""
Otto product classification 
"""


from qsprLib import *
import pandas as pd
from sklearn import preprocessing
from sklearn.lda import LDA
from sklearn.qda import QDA
from pandas.tools.plotting import scatter_matrix

from xgboost_sklearn import *
import xgboost as xgb

#from OneHotEncoder import *

from lasagne_tools import *
from keras_tools import *


def analyzeDataset(Xtrain,Xtest,ytrain):
    plt.hist(ytrain,bins=9)
    plt.show()
    #Xtrain.iloc[:5].hist(color='b', alpha=0.5, bins=50)
    Xtrain.iloc[:,:30].hist(color='b', alpha=1.0, bins=20)
    #scatter_matrix(Xtrain.iloc[:5], alpha=0.2, figsize=(6, 6), diagonal='hist')
    plt.show()
    #pcAnalysis(Xtrain,Xtest,None,None,ncomp=2,transform=False,classification=False)
    #for col in Xtrain.columns:
#	print "Column:",col
#	print Xtrain[col].describe()
#	raw_input()


def prepareDataset(seed=123,nsamples=-1,standardize=False,featureHashing=False,polynomialFeatures=None,OneHotEncoding=False,featureFilter=None,final_filter=None,call_group_data=False,addNoiseColumns=None,log_transform=None,sqrt_transform=False,addFeatures=False,doSVD=False,binning=False,analyzeIt=False,loadFeatures=None,separateClass=False):
  np.random.seed(seed)
  # import data
  Xtrain = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/otto/train.csv').reset_index(drop=True)
  Xtest = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/otto/test.csv')
  #sample = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/otto/sampleSubmission.csv')

  # drop ids and get labels
  labels = Xtrain.target.values
  Xtrain = Xtrain.drop('id', axis=1)
  Xtrain = Xtrain.drop('target', axis=1).astype(np.int32)
  Xtest = Xtest.drop('id', axis=1).astype(np.int32)
  ytrain = preprocessing.LabelEncoder().fit_transform(labels)
  
  if nsamples != -1:
      if 'shuffle' in nsamples: 
	  rows = np.random.choice(len(Xtrain.index), size=len(Xtrain.index),replace=False)
      else:
	  rows = np.random.randint(0,len(Xtrain.index), nsamples,replace=False)
      print "unique: %6.2f"%(float(np.unique(rows).shape[0])/float(rows.shape[0]))
      Xtrain = Xtrain.iloc[rows,:]
      ytrain = ytrain[rows]
      
  #Xtrain = Xall[len(Xtest.index):]
  #Xtest = Xall[:len(Xtest.index)]
  
  # encode labels 
  
  if separateClass is not None and separateClass is not False:
     print "Separating class",separateClass
     print ytrain
     ytrain = (ytrain==separateClass)
     print separateClass
     print ytrain
  
  if analyzeIt:
    analyzeDataset(Xtrain,Xtest,ytrain)
    sys.exit(1)
   
   
   
  Xall = pd.concat([Xtest, Xtrain],ignore_index=True)
  
  if featureFilter is not None:
	print "Using featurefilter..."
	Xall=Xall[featureFilter]
  
  if binning is not None and binning is not False:
	print "Binning data..."
	Xall = data_binning(Xall,binning) 

  if call_group_data:
	print "Group data...",
	print density(Xall)	
	#Xall=pd.DataFrame(group_data2(Xall.values))
	Xall=pd.DataFrame(group_data(Xall))
	#Xall=sparse.csc_matrix(Xall.values)
	print "...new shape:",Xall.shape
	print Xall.describe()
	print density(Xall)

  if polynomialFeatures is not None and polynomialFeatures is not False:
      print "Polynomial feature of degree:",polynomialFeatures
      if  isinstance(polynomialFeatures,str) and 'load' in polynomialFeatures:
	  X_poly = pd.read_csv('poly.csv').reset_index(drop=True)
	  print X_poly.describe()
      else:
	  X_poly = make_polynomials(Xall)
	  X_poly.to_csv('poly.csv')
	  
      if isinstance(polynomialFeatures,list):
	  X_poly = X_poly[polynomialFeatures]
      
      Xall = pd.concat([Xall, X_poly],axis=1)
      #print Xall.describe()
      print "...",Xall.shape
      

  
  if addNoiseColumns is False or addNoiseColumns is not None:
	print "Adding %d random noise columns"%(addNoiseColumns)
	Xrnd = pd.DataFrame(np.random.randn(Xall.shape[0],addNoiseColumns))
	#print "Xrnd:",Xrnd.shape
	#print Xrnd
	for col in Xrnd.columns:
	    Xrnd=Xrnd.rename(columns = {col:'rnd'+str(col+1)})
	
	Xall = pd.concat([Xall, Xrnd],axis=1)
  
  if addFeatures:
    #mad,rank
      print "Additional columns"
      Xall_orig = Xall.copy()
      Xall['row_sum'] = Xall_orig.sum(axis=1)
      Xall['row_median'] = Xall_orig.median(axis=1)
      Xall['row_max'] = Xall_orig.max(axis=1)
      #Xall['row_min'] = Xall_orig.min(axis=1)
      #Xall['row_mean'] = Xall_orig.mean(axis=1)
      #Xall['row_kurtosis'] = Xall_orig.kurtosis(axis=1)
      Xall['row_mad'] = Xall_orig.mad(axis=1)
      
      Xall['arg_max'] = pd.DataFrame(Xall_orig.values).idxmax(axis=1)
      #print Xall['arg_max']
      Xall['arg_min'] = pd.DataFrame(Xall_orig.values).idxmin(axis=1)
      #print Xall['arg_min']
      Xall['non_null'] = (Xall_orig != 0).astype(int).sum(axis=1)
      Xall['row_sd'] = Xall_orig.std(axis=1)
      #Xall['row_prod'] = Xall.prod(axis=1)
      #Xall['feat_11+feat_60'] = (Xall['feat_11'] +Xall['feat_60'])/2
      #Xall['feat_11xfeat_60'] = (Xall['feat_11'] *Xall['feat_60'])/2

      #print Xall.loc[:,['non-null']].describe()
      print Xall.iloc[:,-10:].describe()
      #print Xall.loc[1:5,['sum_counts']]
      #raw_input()

  #indices!!
  if isinstance(loadFeatures,list):
      for name in loadFeatures:
	X_temp = pd.read_csv(X_temp).reset_index(drop=True)
	print X_temp.describe()
      
  if final_filter is not None or final_filter is not None:
      print "Using final_filter..."
      Xall=Xall[final_filter]
  
  if OneHotEncoding:
      #Xtrain = Xall[len(Xtest.index):].values
      #Xtest = Xall[:len(Xtest.index)].values
      encoder = OneHotEncoder()    
      Xall_sparse = encoder.fit_transform(Xall)
      Xall = Xall_sparse

      print "One-hot-encoding...new shape:",Xall.shape
      print type(Xall)
      Xall = Xall.tocsr()
      print density(Xall)
  
  if featureHashing:
      #Xtrain = Xall[len(Xtest.index):]
      #Xtest = Xall[:len(Xtest.index)]
      print "Feature hashing...",#Feature hashing not necessary
      encoder = FeatureHasher(n_features=2**10,dtype=np.int32)
      print encoder
      #encoder = DictVectorizer()#basically one-hot-encoding
      #encoder = OneHotEncoder()
      all_as_dicts = [dict(row.iteritems()) for _, row in Xall.iterrows()]
      #all_as_dicts = [dict(row.iteritems()) for row in Xall.values]
      #print train_as_dicts
      #train_as_dicts = [dict(r.iteritems()) for _, r in Xtrain.iterrows()]  #feature hasher
      Xall_sparse = encoder.fit_transform(all_as_dicts)
      #test_as_dicts = [dict(r.iteritems()) for _, r in Xtest.applymap(str).iterrows()]
      #test_as_dicts = [dict(r.iteritems()) for _, r in Xtest.iterrows()]#feature hasher
      #Xtest_sparse = encoder.transform(test_as_dicts)
      print type(Xall_sparse)
      Xall = Xall_sparse
      #Xtest = Xtest_sparse
      
      #Xall = np.vstack((Xtest,Xtrain))
      print "...new shape:",Xall.shape
      print density(Xall)
  
  
  if log_transform:
	print "log_transform"
	Xall = Xall + 1.0
	Xall=Xall.apply(np.log)
	print Xall.describe()
  
  if sqrt_transform:
        print "sqrt transform"
        Xall = Xall + 3.0/8.0
        Xall=Xall.apply(np.sqrt)
        print Xall.describe()
        
  if doSVD is not None and doSVD is not False:
      print "SVD...components:",doSVD
      #tsvd=TruncatedSVD(n_components=doSVD, algorithm='randomized', n_iter=5, tol=0.0)
      tsvd = RandomizedPCA(n_components=doSVD, whiten=True)
      print tsvd
      Xall=tsvd.fit_transform(Xall.values)
      Xall = pd.DataFrame(Xall)
      print Xall.describe()
      #df_info(Xall)
  
  if standardize:
    Xall = scaleData(lXs=Xall,lXs_test=None)
  

  if isinstance(Xall,pd.DataFrame): Xall = removeLowVariance(Xall,1E-1)
  #Xall = removeCorrelations(Xall,0.99)
  
  
  Xtrain = Xall[len(Xtest.index):].astype(np.float32)
  Xtest = Xall[:len(Xtest.index)].astype(np.float32)
  ytrain = ytrain.astype(np.int32)
  
  #analyzeDataset(Xtrain,Xtest,ytrain)
  
  print "#Xtrain:",Xtrain.shape
    
  
  #if isinstance(Xtest,pd.DataFrame): print Xtest.describe()
  
  if isinstance(Xtrain,pd.DataFrame):
    df_info(Xtrain)
    df_info(ytrain)
    print Xtrain.describe()
    print Xtrain.columns
  
  print "\n#Xtest:",Xtest.shape
  
  return (Xtrain,ytrain,Xtest,labels)

def group_data(data, degree=2):
  """
  Using groupby pandas
  """
  new_data = []
  m,n = data.shape
  for indices in itertools.combinations(range(n), degree):
    tmp=list(data.columns[list(indices)])
    #print tmp
    tmp2 = data.groupby(tmp)
    #print tmp2.describe()
    group_ids = tmp2.grouper.group_info[0]
    #print group_ids
    #raw_input()
    new_data.append(group_ids)
  return np.array(new_data).T


def dict_encode(encoding, value):
    if not value in encoding:
        encoding[value] = {'code': len(encoding)+1, 'count': 0}
    enc = encoding[value]
    enc['count'] += 1
    encoding[value] = enc


def dict_decode(encoding, value, min_occurs):
    enc = encoding[value]
    if enc['count'] < min_occurs:
        return -1
    else:
        return enc['code']


def group_data2(data, degree=2, min_occurs=2):
    """ 
    Group data using min_occurs
    
    Groups all columns of data into all combinations of degree
    """
    m, n = data.shape
    encoding = dict()
    for indexes in itertools.combinations(range(n), degree):
        for v in data[:, indexes]:
            dict_encode(encoding, tuple(v))
    new_data = []
    for indexes in itertools.combinations(range(n), degree):
        new_data.append([dict_decode(encoding, tuple(v), min_occurs) for v in data[:, indexes]])
    return np.array(new_data).T


def group_data3(data, degree=3, cutoff = 1, hash=hash):
    """ 
    Luca Massaron
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    
    new_data = []
    m,n = data.shape
    for indexes in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indexes]])
    for z in range(len(new_data)):
        counts = dict()
        useful = dict()
        for item in new_data[z]:
            if item in counts:
                counts[item] += 1
                if counts[item] > cutoff:
                    useful[item] = 1
            else:
                counts[item] = 1
        for j in range(len(new_data[z])):
            if not new_data[z][j] in useful:
                new_data[z][j] = 0
    return np.array(new_data).T


def makePredictions(model=None,Xtest=None,filename='submission.csv'):
    sample = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/otto/sampleSubmission.csv')
    if model is not None: 
	preds = model.predict_proba(Xtest)
    else:
	preds = Xtest
    # create submission file
    preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
    preds.to_csv(filename, index_label='id')
    
#TODO check sklearn 1.6 with class_weight crossvalidation
#https://github.com/cudamat/cudamat
#TODO optimize logloss directly http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf in xgboost_sklearn
#sparse data with LinearSVC
#https://www.kaggle.com/c/tradeshift-text-classification/forums/t/10537/beat-the-benchmark-with-less-than-400mb-of-memory
#http://blogs.technet.com/b/machinelearning/archive/2014/09/24/online-learning-and-sub-linear-debugging.aspx
#new columns: http://stackoverflow.com/questions/16139147/conditionally-combine-columns-in-pandas-data-frame
#group data versus interaction!
# use MKL with numpy https://gehrcke.de/2014/02/building-numpy-and-scipy-with-intel-compilers-and-intel-mkl-on-a-64-bit-machine/
#use greedy algorithm with RF to select new features->ALL features important
#sklearn.multiclass.OneVsOneClassifier(estimator, n_jobs=1 with logistc reg
#http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping
#http://pyevolve.sourceforge.net/0_6rc1/getstarted.html
#leave one out encoding of categorical data: http://nycdatascience.com/news/featured-talk-1-kaggle-data-scientist-owen-zhang/
# weights of RF, give out LL for each class
#Baggging of Neural Nets!
#Reduce variance reduction within  ensemble creation!??
#DataFrame(randn(1000000,20)).to_hdf('test.h5','df',complevel=9,complib='blosc')
#sampleweights for classes 0,1, 3
#large decay rates and regularization
#http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html#example-calibration-plot-calibration-multiclass-py
#ensemble with hard voting -> calibration->NO
#shuffle split for ensemble building!!...
#interactions: http://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/13486/variable-interaction->OK
#tune subsample for XGB->OK
# one versus all with pipeline i.e. feature selection!?
#tune gamma ->NO
#ensembling: aggregation via argmax, then voting or one hote encoding and log reg 
#ensembling: use gradient boosting for building the ensemble...!!!YES
#use RF with SVD? --->NO??
#look at accuracies for OneVsOneClassifier->no predict_proba
#modify objective for xgboost for taking care of class 0,1 and 3?
#bag xgb with colsample_bytree ->OK
#sample weights ...
#use xgboost one vs one for ensemling only as metafeature ->OK
#http://stackoverflow.com/questions/19575348/tricks-to-make-an-aws-spot-instance-persistent
#feature rank + feature quantiles
# add ovo as feature first!!!
# calibration with ensemble!!!->NOVO
# use early stopping!!

if __name__=="__main__":
    """   
    MAIN PART
    """ 
    # Set a seed for consistant results
    t0 = time()
    
    #pd.set_option('display.height', 1000)
    #pd.set_option('display.max_rows', 500)
    #pd.set_option('display.max_columns', 500)
    #pd.set_option('display.width', 1000)
    print "numpy:",np.__version__
    print "pandas:",pd.__version__
    print "scipy:",sp.__version__
    #import sklearn
    #print "sklearn:",sklearn.__version__
    #after linear feature selection, features are orderd according to their effect
    ordered92=['feat_11', 'feat_60', 'feat_34', 'feat_14', 'feat_90', 'feat_15', 'feat_62', 'feat_42', 'feat_39', 'feat_36', 'feat_75', 'feat_68', 'feat_9', 'feat_43', 'feat_40', 'feat_76', 'feat_86', 'feat_26', 'feat_35', 'feat_59', 'feat_47', 'feat_17', 'feat_48', 'feat_69', 'feat_50', 'feat_91', 'feat_92', 'feat_56', 'feat_53', 'feat_25', 'feat_84', 'feat_57', 'feat_78', 'feat_58', 'feat_41', 'feat_32', 'feat_67', 'feat_72', 'feat_77', 'feat_64', 'feat_20', 'feat_71', 'feat_83', 'feat_19', 'feat_23', 'feat_88', 'feat_33', 'feat_73', 'feat_93', 'feat_3', 'feat_81', 'feat_13', 'feat_6', 'feat_31', 'feat_52', 'feat_4', 'feat_82', 'feat_51', 'feat_28', 'feat_2', 'feat_12', 'feat_21', 'feat_80', 'feat_49', 'feat_54', 'feat_65', 'feat_5', 'feat_63', 'feat_46', 'feat_27', 'feat_44', 'feat_55', 'feat_7', 'feat_61', 'feat_70', 'feat_10', 'feat_18', 'feat_22', 'feat_38', 'feat_8', 'feat_89', 'feat_16', 'feat_66', 'feat_45', 'feat_30', 'feat_79', 'feat_1', 'feat_24', 'feat_74', 'feat_87', 'feat_37', 'feat_29']
    all_features=[u'feat_1', u'feat_2', u'feat_3', u'feat_4', u'feat_5', u'feat_6', u'feat_7', u'feat_8', u'feat_9', u'feat_10', u'feat_11', u'feat_12', u'feat_13', u'feat_14', u'feat_15', u'feat_16', u'feat_17', u'feat_18', u'feat_19', u'feat_20', u'feat_21', u'feat_22', u'feat_23', u'feat_24', u'feat_25', u'feat_26', u'feat_27', u'feat_28', u'feat_29', u'feat_30', u'feat_31', u'feat_32', u'feat_33', u'feat_34', u'feat_35', u'feat_36', u'feat_37', u'feat_38', u'feat_39', u'feat_40', u'feat_41', u'feat_42', u'feat_43', u'feat_44', u'feat_45', u'feat_46', u'feat_47', u'feat_48', u'feat_49', u'feat_50', u'feat_51', u'feat_52', u'feat_53', u'feat_54', u'feat_55', u'feat_56', u'feat_57', u'feat_58', u'feat_59', u'feat_60', u'feat_61', u'feat_62', u'feat_63', u'feat_64', u'feat_65', u'feat_66', u'feat_67', u'feat_68', u'feat_69', u'feat_70', u'feat_71', u'feat_72', u'feat_73', u'feat_74', u'feat_75', u'feat_76', u'feat_77', u'feat_78', u'feat_79', u'feat_80', u'feat_81', u'feat_82', u'feat_83', u'feat_84', u'feat_85', u'feat_86', u'feat_87', u'feat_88', u'feat_89', u'feat_90', u'feat_91', u'feat_92', u'feat_93']
    start_set = ['feat_11', 'feat_60', 'feat_34', 'feat_14', 'feat_90', 'feat_15', 'feat_62', 'feat_42', 'feat_39', 'feat_36', 'feat_75', 'feat_68', 'feat_9', 'feat_43', 'feat_40', 'feat_76', 'feat_86', 'feat_26', 'feat_35', 'feat_59', 'feat_47', 'feat_17', 'feat_48', 'feat_69', 'feat_50', 'feat_91', 'feat_92', 'feat_56', 'feat_53', 'feat_25', 'feat_84', 'feat_57', 'feat_78', 'feat_58', 'feat_41', 'feat_32', 'feat_67', 'feat_72', 'feat_77', 'feat_64', 'feat_20', 'feat_71', 'feat_83', 'feat_19', 'feat_23', 'feat_88', 'feat_33', 'feat_73', 'feat_93', 'feat_3', 'feat_81', 'feat_13']
    ga_set=[u'feat_1', u'feat_3', u'feat_9', u'feat_10', u'feat_11', u'feat_12', u'feat_13', u'feat_14', u'feat_15', u'feat_16', u'feat_18', u'feat_23', u'feat_24', u'feat_25', u'feat_26', u'feat_27', u'feat_28', u'feat_32', u'feat_33', u'feat_34', u'feat_35', u'feat_36', u'feat_39', u'feat_40', u'feat_41', u'feat_42', u'feat_43', u'feat_45', u'feat_47', u'feat_48', u'feat_49', u'feat_53', u'feat_56', u'feat_57', u'feat_59', u'feat_60', u'feat_62', u'feat_63', u'feat_64', u'feat_67', u'feat_68', u'feat_69', u'feat_71', u'feat_72', u'feat_73', u'feat_77', u'feat_79', u'feat_80', u'feat_81', u'feat_84', u'feat_86', u'feat_88', u'feat_90', u'feat_92', u'rnd2', u'arg_max', u'row_sd']    
    interactions=['feat_34xfeat_83','feat_42xfeat_26', 'feat_34xfeat_48', 'feat_9xfeat_67','feat_60xfeat_43','feat_34xfeat_43','feat_11xfeat_64','feat_34xfeat_25','feat_60xfeat_15']   
    addedFeatures=[u'row_sum', u'row_median', u'row_max', u'row_mad', u'arg_max', u'arg_min', u'non_null', u'row_sd']
    addedFeatures_short=[u'arg_max', u'row_sd']
    addedFeatures_best=[u'row_median',u'arg_max',u'row_max',u'non_null',u'arg_min']
    
    #NNsetting
    #"""
    nsamples='shuffle'#61878
    standardize=False
    polynomialFeatures=False#'load'
    featureHashing=False
    OneHotEncoding=True
    analyzeIt=False
    call_group_data=False
    log_transform=False
    sqrt_transform=False
    addNoiseColumns=None
    addFeatures=False
    doSVD=None
    binning=None
    #featureFilter=all_ordered
    separateClass=None
    featureFilter=None#start_set
    final_filter=None#start_set+interactions#all_features+addedFeatures_best
    loadFeatures=None
    

    """
    #Normal settings
    nsamples='shuffle'
    standardize=False
    polynomialFeatures=None#'load'
    featureHashing=False
    OneHotEncoding=False
    analyzeIt=False
    call_group_data=False
    log_transform=False
    addNoiseColumns=None
    addFeatures=False
    doSVD=None
    binning=None
    #featureFilter=all_ordered
    featureFilter=None
    final_filter=None
    """
    
    Xtrain, ytrain, Xtest, labels  = prepareDataset(nsamples=nsamples,standardize=standardize,featureHashing=featureHashing,OneHotEncoding=OneHotEncoding,polynomialFeatures=polynomialFeatures,featureFilter=featureFilter,final_filter=final_filter, call_group_data=call_group_data,log_transform=log_transform,addNoiseColumns=addNoiseColumns,addFeatures=addFeatures,doSVD=doSVD,binning=binning,sqrt_transform=sqrt_transform,analyzeIt=analyzeIt,separateClass=separateClass)
    
    #Xtrain = sparse.csr_matrix(Xtrain)
    print type(Xtrain)
    Xtrain = Xtrain.astype(np.float32)
    #ytrain = ytrain.astype(np.float32)
    #df_info(Xtrain)
    density(Xtrain)

    model = LogisticRegression(C=1.0,penalty='l2')
    #model = LogisticRegression(C=1.0,class_weight=None,penalty='l2',solver='lbfgs', multi_class='ovr' )#0.671
    #model = LogisticRegression(C=1E-1,class_weight=None,penalty='l1',solver='liblinear', multi_class='ovr' )#0.671
    #model = LogisticRegression(C=1E-1,class_weight=None,penalty='l1' )
    #model = Pipeline([('filter', GenericUnivariateSelect(chi2, param=97,mode='percentile')), ('model', LogisticRegression(C=1.0,solver='lbfgs', multi_class='ovr',class_weight='auto'))])
    #model = Pipeline([('filter', GenericUnivariateSelect(f_classif, param=95,mode='percentile')), ('model', LogisticRegression(C=10.0))])
    #model = OneVsRestClassifier(model,n_jobs=1)
    #model = Pipeline([('pca', PCA(n_components=20)),('model', LogisticRegression(C=1.0))])

    #model = SGDClassifier(alpha=1E-6,n_iter=250,shuffle=True,loss='log',penalty='l2',n_jobs=1,learning_rate='optimal',verbose=False)
    #model = SGDClassifier(alpha=1E-6,n_iter=800,shuffle=True,loss='modified_huber',penalty='l2',n_jobs=8,learning_rate='optimal',verbose=False)#mll=0.68
    #model =  RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=1,n_jobs=4,criterion='gini', max_features=20)
    #model = CalibratedClassifierCV(model, method='isotonic', cv=3)
    #model = KNeighborsClassifier(n_neighbors=5)
    #model =  RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=1,n_jobs=1,criterion='gini', max_features=20,oob_score=False,class_weight='auto')
    #model =  RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=1,n_jobs=1,criterion='gini', max_features=20,oob_score=False,class_weight=None)
    #model = CalibratedClassifierCV(model, method='isotonic', cv=3)
    
    #model =  ExtraTreesClassifier(bootstrap=False,n_estimators=500,max_depth=None,min_samples_leaf=1,n_jobs=4,criterion='entropy', max_features=20,oob_score=False)
    
    #basemodel = GradientBoostingClassifier(loss='deviance',n_estimators=4, learning_rate=0.03, max_depth=10,subsample=.5,verbose=1)
    
    #model = SVC(kernel='rbf',C=10.0, gamma=0.0, verbose = 0, probability=True)
    
    #model = XgboostClassifier(booster='gblinear',n_estimators=50,alpha_L1=0.1,lambda_L2=0.1,n_jobs=2,objective='multi:softprob',eval_metric='mlogloss',silent=1)#0.63
    #model = XgboostClassifier(n_estimators=400,learning_rate=0.05,max_depth=10,subsample=.5,n_jobs=1,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1,eval_size=-1)#0.45
    #basemodel1 = XgboostClassifier(n_estimators=120,learning_rate=0.1,max_depth=6,subsample=.5,n_jobs=1,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1,eval_size=0.0)#0.46
    #basemodel1 = LogisticRegression(C=1.0,penalty='l2')
    #model = OneVsOneClassifier(basemodel1)
    
    #hyper_opt1 0.443
    #basemodel1 = XgboostClassifier(n_estimators=200,learning_rate=0.13,max_depth=10,subsample=.82,colsample_bytree=0.56,n_jobs=4,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1)
    #model = BaggingClassifier(base_estimator=basemodel1,n_estimators=10,n_jobs=1,verbose=2,random_state=None,max_samples=1.0,max_features=0.94,bootstrap=False)#for some reason parallelization does not work...?estimated 12h runs 10 bagging iterations with 400 trees in 8fold crossvalidation

    #hyper_opt2 0.446
    #basemodel1 = XgboostClassifier(n_estimators=200,learning_rate=0.166,max_depth=11,subsample=.83,colsample_bytree=0.58,n_jobs=4,objective='multi:softprob',eval_metric='mlogloss',booster='gbtree',silent=1)
    #model = BaggingClassifier(base_estimator=basemodel1,n_estimators=10,n_jobs=1,verbose=2,random_state=None,max_samples=0.93,max_features=0.92,bootstrap=False)

    #model = nnet9
    #model = KerasNN(dims=93,nb_classes=9,nb_epoch=60,learning_rate=0.015,validation_split=0.15,batch_size=64,verbose=1)
    #model = KerasNN3(dims=93,nb_classes=9,nb_epoch=100,learning_rate=0.004,validation_split=0.0,batch_size=128,verbose=1)
    #print ytrain.shape

    #with open('nnet1.pickle', 'rb') as f:  # !
    #        net_pretrain = pickle.load(f)  # !
    #        model.load_weights_from(net_pretrain)
    #
    #"""
    #model = BaggingClassifier(base_estimator=model,n_estimators=2,n_jobs=1,verbose=2,random_state=None,max_samples=1.0,max_features=1.0,bootstrap=False)
    
    scoring_func = make_scorer(multiclass_log_loss, greater_is_better=False, needs_proba=True)
    #scoring_func = make_scorer(accuracy_score, greater_is_better=True, needs_proba=False)
    #analyzeLearningCurve(model,Xtrain,ytrain,cv=StratifiedShuffleSplit(ytrain,24,test_size=0.125),score_func=scoring_func)
    #model = buildClassificationModel(model,Xtrain,ytrain,list(set(labels)).sort(),trainFull=False,cv=StratifiedKFold(ytrain,8,shuffle=True))
    model = buildModel(model,Xtrain,ytrain,cv=StratifiedKFold(ytrain,8,shuffle=True),scoring=scoring_func,n_jobs=8,trainFull=False,verbose=True)
    #model = buildModel(model,Xtrain,ytrain,cv=StratifiedShuffleSplit(ytrain,2,test_size=0.01),scoring=scoring_func,n_jobs=1,trainFull=False,verbose=True)
    #model = buildClassificationModel(model,Xtrain,ytrain,list(set(labels)).sort(),trainFull=False,cv=StratifiedShuffleSplit(ytrain,4,test_size=0.001))
    
    #model.fit(Xtrain.values, ytrain)
    #with open('nnet1.pickle', 'wb') as f:
    #  pickle.dump(model, f, -1)
   
    
    
    #plotNN(model)
    
    #genetic_feature_selection(model,Xtrain,ytrain,Xtest,pool_features=None,start_features=ga_set,scoring_func=scoring_func,cv=StratifiedShuffleSplit(ytrain,2,test_size=0.2),n_iter=10,n_pop=20,n_jobs=2)
    #genetic_feature_selection(model,Xtrain,ytrain,Xtest,pool_features=None,start_features=start_set,scoring_func=scoring_func,cv=StratifiedKFold(ytrain,4,shuffle=True),n_iter=10,n_pop=20,n_jobs=4)
    #iterativeFeatureSelection(model,Xtrain,Xtest,ytrain,iterations=1,nrfeats=1,scoring=scoring_func,cv=StratifiedKFold(ytrain,5),n_jobs=1)
    #greedyFeatureSelection(model,Xtrain,ytrain,itermax=40,itermin=30,pool_features=Xtrain.iloc[:,93:].columns ,start_features=None,verbose=True, cv=StratifiedKFold(ytrain,8,shuffle=True), n_jobs=8,scoring_func=scoring_func)
    
    #model.fit(Xtrain,ytrain)
    #parameters = {'n_estimators':[120],'max_depth':[6],'learning_rate':[0.1],'subsample':[0.5],'colsample_bytree':[1.0],'gamma':[0.0,.5,1.0],'min_child_weight':[1,2]}
    #parameters = {'hidden1_num_units': [500,600,700],'dropout1_p':[0.5],'hidden2_num_units': [500,600],'dropout2_p':[0.5],'hidden3_num_units': [500,300],'max_epochs':[75,100]}
    #parameters = {'hidden1_num_units': [600],'dropout1_p':[0.5],'hidden2_num_units': [600],'dropout2_p':[0.5],'hidden3_num_units': [600],'objective_alpha':[1E-9,0.000001,0.00005],'max_epochs':[50,60,70]}#Lasagne
    #parameters = {'hidden1_num_units': [300,600],'dropout1_p':[0.5],'maxout1_ds':[2,3],'hidden2_num_units': [300,600],'dropout2_p':[0.0,0.25,0.5],'maxout2_ds':[2,3],'hidden3_num_units': [300,600],'dropout3_p':[0.0,0.25,0.5],'maxout3_ds':[2,3],'max_epochs':[75,150]}
    #parameters = {'dropout0_p':[0.05,0.1,0.15],'hidden1_num_units': [900],'dropout1_p':[0.5],'hidden2_num_units': [500],'dropout2_p':[0.5,0.4,0.25],'hidden3_num_units': [250],'dropout3_p':[0.5,0.4,0.25],'max_epochs':[100]}
    #model = makeGridSearch(model,Xtrain,ytrain,n_jobs=1,refit=False,cv=StratifiedShuffleSplit(ytrain,4,test_size=0.125),scoring=scoring_func,parameters=parameters,random_iter=10)
    #parameters = {'objective_alpha':[1E-9,1E-7,1E-6,1E-5]}#Lasagne
    #model = makeGridSearch(model,Xtrain,ytrain,n_jobs=8,refit=False,cv=StratifiedKFold(ytrain,5,shuffle=True),scoring=scoring_func,parameters=None,random_iter=-1)
    #makePredictions(model,Xtest,filename='/home/loschen/Desktop/datamining-kaggle/otto/submissions/submission24042015a.csv')
    plt.show()
    print("Model building done in %fs" % (time() - t0))

