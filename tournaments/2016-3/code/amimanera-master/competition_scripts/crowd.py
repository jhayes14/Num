#!/usr/bin/python 
# coding: utf-8

from qsprLib import *

from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search

from sklearn.feature_extraction import text

import re

from nltk import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk import corpus

from crowd_features import *

from lasagne_tools import *

class MyTokenizer(object):
    """
    http://scikit-learn.org/stable/modules/feature_extraction.html
    http://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
    http://nltk.org/api/nltk.tokenize.html
    """
    def __init__(self,stemmer=None,stop_words=None):
      print "Using special tokenizer, stemmer:",stemmer," stop_words:",stop_words
      self.wnl = stemmer
      self.stop_words = stop_words
    def __call__(self, doc): 
      #words=[word_tokenize(t) for t in sent_tokenize(doc)]
      #words=[item for sublist in words for item in sublist]
      #simple tokenizer 2 e.g. scikit learn, preprocessing is done beforehand
      token_pattern = re.compile(r'\w{1,}')
      words = token_pattern.findall(doc)
      #print words
      #raw_input()
      #print "n words, after:",len(words)
      
      if hasattr(self.wnl,'stem'):
	  words=[self.wnl.stem(t) for t in words]

      return words

class MyStemmer(PorterStemmer):
    def stem(self,word):
	return super(MyStemmer,self).stem(word)


def loadData(nsamples=-1):
    # Load the training file
    Xtrain = pd.read_csv('data/train.csv')
    Xtest = pd.read_csv('data/test.csv')
    
    # we dont need ID columns
    idx = Xtest.id.values.astype(int)
    Xtrain = Xtrain.drop('id', axis=1)
    Xtest = Xtest.drop('id', axis=1)
    
    Xtrain["product_description"] = Xtrain["product_description"].fillna("no_description")
    Xtest["product_description"] = Xtest["product_description"].fillna("no_description")
     
    # create labels. drop useless columns
    ytrain = Xtrain.median_relevance.values
    sample_weight = Xtrain.relevance_variance
    #sample_weight.hist()
    #idx=sample_weight>1.0
    #print "idx",idx
    #sample_weight[:] = 1.0
    #sample_weight[idx] = 0.25
    #sample_weight.hist()
    #sample_weight = sample_weight.apply()
    #plt.show()
    Xtrain = Xtrain.drop(['median_relevance', 'relevance_variance'], axis=1)
    
    if nsamples != -1:
      if isinstance(nsamples,str) and 'shuffle' in nsamples:
	  print "Shuffle train data..."
	  rows = np.random.choice(len(Xtrain.index), size=len(Xtrain.index),replace=False)
      else:
	  rows = np.random.randint(0,len(Xtrain.index), nsamples)
      print "unique: %6.2f"%(float(np.unique(rows).shape[0])/float(rows.shape[0]))
      Xtrain = Xtrain.iloc[rows,:]
      ytrain = ytrain[rows]
      sample_weight = sample_weight[rows]

    #create cv labels
    le = preprocessing.LabelEncoder()
    labels = pd.DataFrame(le.fit_transform(Xtrain['query'].values),columns=['labels'])
    labels.to_csv('unique_queries.csv')
    labels = pd.DataFrame(le.fit_transform(Xtrain['product_title'].values),columns=['labels'])
    labels.to_csv('unique_titles.csv')
    
    
    return Xtest,Xtrain,ytrain,idx,sample_weight


def prepareDataset(seed=123,nsamples=-1,cleanse=None,useOnlyTrain=False,vectorizer=None,doSVD=None,standardize=False,doBenchMark=False,computeFeatures=None,computeWord2Vec=None,vectorizeFirstOnly=False,computeSynonyma=None,computeKaggleDistance=None,computeKaggleTopics=None,addNoiseColumns=None,doSeparateTFID=None,doSVDseparate=False,doSVDseparate_2nd=None,computeSim=None,stop_words=corpus.stopwords.words('english'),doTFID=False,concat=False,useAll=False,concatTitleDesc=False,doKmeans=False):
    np.random.seed(seed)
    
    Xtest,Xtrain,ytrain,idx,sample_weight = loadData(nsamples=nsamples)
    Xall = pd.concat([Xtest, Xtrain])
    print "Original shape:",Xall.shape

    if cleanse is not None:
	if isinstance(cleanse,str):
	  print "Loading cleansed data..."
	  Xall = pd.read_csv('Xall_cleansed.csv',index_col=0)
	else:
	  Xall = cleanse_data(Xall)
	  Xall.to_csv("Xall_cleansed.csv")
	
    
    if computeSynonyma is not None:
	print Xall['query'].head(10)
	Xall = makeQuerySynonyms(Xall)
	print Xall['query'].head(10)
    
    if concatTitleDesc:
	print "Concatenating title+description..."
	Xall['product_title'] = Xall.apply(lambda x:'%s %s' % (x['product_title'],x['product_description']),axis=1)
	Xall = Xall.drop(['product_description'], axis=1)
	print Xall.head(10)
    
    if doBenchMark:
	Xall = useBenchmarkMethod(Xall)
    
    print "computeFeatures:",computeFeatures
    Xfeat = None
    if computeFeatures is not None:
	if isinstance(computeFeatures,str):
	  print "Loading features data..."
	  Xfeat = pd.read_csv('Xfeat.csv',index_col=0)
	else:
	  Xfeat = additionalFeatures(Xall,verbose=False)
	  Xfeat.to_csv("Xfeat.csv")
        
        print Xfeat.describe()

    Xw2vec = None
    if computeWord2Vec is not None:
	if isinstance(computeWord2Vec,str):
	  print "Loading word2vec features..."
	  Xw2vec = pd.read_csv('Xw2vec.csv',index_col=0)
	else:
	  Xw2vec = genWord2VecFeatures(Xall,verbose=False)
	  Xw2vec.to_csv("Xw2vec.csv")      
        print Xw2vec.describe()
	


    Xkdist = None
    if computeKaggleDistance is not None:
	Xkdist = createKaggleDist(Xall,general_topics=computeKaggleTopics,verbose=False)

    Xsim = None
    if computeSim is not None:
	Xsim = computeSimilarityFeatures(Xall,columns=['query','product_title'],verbose=False,useOnlyTrain=useOnlyTrain,startidx=len(Xtest.index),stop_words=stop_words)
	print Xsim.describe()
    
    if doSeparateTFID is False or doSeparateTFID is not None:
	analyze=False
	print "Vectorize columns separately...",

	tokenizer = MyTokenizer(stemmer=PorterStemmer())
	
	#vectorizer = HashingVectorizer(stop_words=stop_words,ngram_range=(1,2),analyzer="char", non_negative=True, norm='l2', n_features=2**18)
	#vectorizer = CountVectorizer(min_df=3,  max_features=None, lowercase=True,analyzer="word",ngram_range=(1,2),stop_words=stop_words,strip_accents='unicode')
	#vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2')
	if vectorizer is None:
	  print "Using default vectorizer..."
	  vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 5), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2')
	else:
	  print "Using vectorizer:"
	  print vectorizer
	Xtrain = Xall[len(Xtest.index):]
	Xtest_t = Xall[:len(Xtest.index)]
	
	for i,col in enumerate(doSeparateTFID):
	    print "Vectorizing: ",col
	    
	    print "Is null:",Xtrain[col].isnull().sum()
	    print Xtrain[col].describe()
	    
	    if i>0:
		if not vectorizeFirstOnly:
		    print "Vecorizing col:",col
		    if useOnlyTrain:
			print "Using only training data for TFIDF."
			vectorizer.fit(Xtrain[col])
		    else:
			vectorizer.fit(Xall[col])#should reduce overfitting-> padded with fake data!!
		else:
		    print "Only transform for col:",col
		Xs_all_new = vectorizer.transform(Xall[col])
		print "Shape Xs_all 2nd after vectorization:",Xs_all_new.shape
		
		if doSVDseparate:
		    if doSVDseparate_2nd is None:
		      n_components = doSVDseparate
		    else:
		      n_components = doSVDseparate_2nd
		    print "col: %s n_components: %d"%(col,n_components)
		    reducer=TruncatedSVD(n_components=n_components, algorithm='randomized', n_iter=5, tol=0.0)
		    #Xs_all_new=sparse.vstack((Xs_test_new,Xs_train_new))
		    Xs_all_new=reducer.fit_transform(Xs_all_new)
		    Xs_all = pd.DataFrame(np.hstack((Xs_all,Xs_all_new)))
		    print "Shape Xs_all after SVD:",Xs_all.shape
		
	    else:
		#we fit first column on all data!!!
		if useOnlyTrain:
		  print "Using only training data for TFIDF."
		  vectorizer.fit(Xtrain[col])
		else:
		  vectorizer.fit(Xall[col])#should reduce overfitting?
		Xs_all = vectorizer.transform(Xall[col])
		print "Shape Xs_all after vectorization:",Xs_all.shape
		
		if doSVDseparate is not None:
		    n_components = doSVDseparate
		    print "col: %s n_components: %d"%(col,n_components)
		    reducer=TruncatedSVD(n_components=n_components, algorithm='randomized', n_iter=5, tol=0.0)
		    #Xs_all=sparse.vstack((Xs_test,Xs_train))
		    Xs_all=reducer.fit_transform(Xs_all)
		    print "Shape Xs_all after SVD:",Xs_all.shape
	
	    if analyze:
		analyzer(vectorizer,Xs_train, ytrain)
	
	
	if doSVDseparate:
	    #Xall = pd.concat([Xs_test, Xs_train])
	    Xall = Xs_all
	
	print type(Xall)
	print " shape:",Xall.shape
	#vectorizer.get_features_names()[0:2]

    if concat:
	print "Concatenating query+title, discarding description..."
	# do some lambda magic on text columns
	#data = list(Xall.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
	#print data[0]
	Xall['query'] = Xall.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1)
	Xall = Xall.drop(['product_title','product_description'], axis=1)
	print Xall.head(10)
    
    

    if doTFID:
	print "Fit TFIDF..."
	stop_words = text.ENGLISH_STOP_WORDS
	
	if vectorizer is None: 
	  print "Default vectorizer:"
	  vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 
		strip_accents='unicode', analyzer='word',
		ngram_range=(1, 5), use_idf=True,smooth_idf=True,sublinear_tf=True,
		stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2',tokenizer=None)
	
	print vectorizer
	if useAll:
	    Xall =  vectorizer.fit_transform(Xall['query'])
	    print Xall
	    
	else:
	    Xtrain = Xall[len(Xtest.index):]
	    Xtest_t = Xall[:len(Xtest.index)]
	    Xtrain=vectorizer.fit_transform(Xtrain['query'])
	    Xtest_t=vectorizer.transform(Xtest_t['query'])
	    Xall = sparse.vstack((Xtest_t,Xtrain),format="csr")
	
	#print vectorizer.get_features_names()
	
	print "Xall:"
	print Xall.shape

    reducer=None
    if doKmeans is not None and doKmeans is not False:
	print "Kmeans components..."
	reducer = MiniBatchKMeans(init='k-means++', n_clusters=doKmeans, n_init=3,batch_size=400)
    
    if doSVD is not None and doSVD is not False:
	print "SVD...components:",doSVD
	reducer=TruncatedSVD(n_components=doSVD, algorithm='randomized', n_iter=5, tol=0.0)
	#reducer = RandomizedPCA(n_components=doSVD, whiten=True)
	
    if reducer is not None:
	print reducer
	
	if useAll:
	  Xall = reducer.fit_transform(Xall)
	  
	else:
	  Xtrain = Xall[len(Xtest.index):]
	  Xtest_t = Xall[:len(Xtest.index)]
	  Xtrain=reducer.fit_transform(Xtrain)
	  Xtest_t=reducer.transform(Xtest_t)
	  Xall = np.vstack((Xtest_t,Xtrain))
	
	Xall = pd.DataFrame(Xall)
    
    """
    if computeSim:
	Xsim = computeSimilarityFeatures(vectorizer=None,nsamples=nsamples,stop_words=stop_words)
      
	if not isinstance(Xall,pd.DataFrame):
	    print "X is not a DataFrame, converting from,",type(Xall)
	    Xall = pd.DataFrame(Xall.todense())
	
	Xall = pd.concat([Xall,Xsim], axis=1)
	print Xall.describe()
	print Xall.columns[-5:]
    """

    if addNoiseColumns is not None:
	print "Adding %d random noise columns"%(addNoiseColumns)
	Xrnd = pd.DataFrame(np.random.randn(Xall.shape[0],addNoiseColumns))
	#print "Xrnd:",Xrnd.shape
	#print Xrnd
	for col in Xrnd.columns:
	    Xrnd=Xrnd.rename(columns = {col:'rnd'+str(col+1)})
	
	Xall = pd.concat([Xall, Xrnd],axis=1)

    if Xsim is not None:
	  if isinstance(computeSim,str) and 'only' in computeSim:
	      Xall = Xsim
	  else:
	      Xall = pd.concat([Xall,Xsim], axis=1)

    if Xfeat is not None:
	  if isinstance(computeFeatures,str) and 'only' in computeFeatures:
	      Xall = Xfeat
	  else:
	      Xall = pd.concat([Xall,Xfeat], axis=1)
    
    if Xkdist is not None:
	Xall = pd.concat([Xall,Xkdist], axis=1)


    if Xw2vec is not None:
	Xall = pd.concat([Xall,Xw2vec], axis=1)

    if standardize:
	if not isinstance(Xall,pd.DataFrame):
	    print "X is not a DataFrame, converting from,",type(Xall)
	    Xall = pd.DataFrame(Xall.todense())
	Xall = scaleData(lXs=Xall,lXs_test=None)
    
    Xtrain = Xall[len(Xtest.index):]
    Xtest = Xall[:len(Xtest.index)]
    
    if not isinstance(Xtrain,list):
      print "#Xtrain:",Xtrain.shape
      print "#Xtest:",Xtest.shape
    
    #print type(ytrain)
    print "#ytrain:",ytrain.shape
    print "#data preparation finished!\n\n"
    return(Xtrain,ytrain,Xtest,idx,sample_weight)



def analyzer(vectorizer,Xs_train, ytrain):
    if isinstance(vectorizer,TfidfVectorizer):
		indices = np.argsort(vectorizer.idf_)[::-1]
		features = vectorizer.get_feature_names()
		top_n = 20
		top_features = [features[i] for i in indices[:top_n]]
		print top_features
	    
    else:
	#indices = np.argsort(vectorizer.idf_)[::-1]
	features = vectorizer.get_feature_names()
	ch2 = SelectKBest(chi2, "all")
	X_train = ch2.fit_transform(Xs_train, ytrain)
	indices = np.argsort(ch2.scores_)[::-1]
	
	for idx in indices:
	    print "idx: %10d Key: %32s  score: %20d"%(idx,features[idx],ch2.scores_[idx])
	    #raw_input()

 
def makePredictions(model=None,Xtest=None,idx=None,filename='submission.csv'):
    # Create your first submission file
    if model is not None: 
	preds = model.predict(Xtest)
	print type(model)
	if isinstance(model,NeuralNet):
	  preds = preds + 1
	
    else:
	preds = Xtest

    if idx is None:
	Xtest = pd.read_csv('data/test.csv')
	idx = Xtest.id.values.astype(int)
    
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv(filename, index=False)


if __name__=="__main__":
    """   
    MAIN PART
    """ 
    #TODO new features from text
    #TODO e.g. length of query, product_title, introduce categories based on titles, use stemmer
    #TODO RandomTreesEmbedding¶
    #https://www.kaggle.com/wliang88/crowdflower-search-relevance/extra-engineered-features-w-svm
    #introduce penalty in classification loss func due to class distance ...? jaccard distance
    #http://stackoverflow.com/questions/17388213/python-string-similarity-with-probability
    #http://avrilomics.blogspot.de/2014/01/calculating-similarity-measure-for-two.html
    #use sample weights...remove high variance queries ->NO
    #use t-sne instead of SVD
    #from nltk.corpus import wordnet as wn
    #compute maximum cosine similarity _>OK
    #use vocabulary from description to transform query...->OK
    #cv: https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/14350/cross-validation-and-leaderboard-score
    #separate SVD for both cases...->OK
    
    #number of words found in title->OK
    #do they have the same ordeR?
    #either unsupervised manipulation to whole dataset to avoid overfitting or incldue manipulation into cv
    #https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/14502/word2vec-doc2vec
    #use similarity total count like in: https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/14159/beating-the-benchmark-yet-again?page=2
    #use NN: https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/14159/beating-the-benchmark-yet-again?page=3
    #cross-validation: https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/14350/cross-validation-and-leaderboard-score
    #for ensemble do like 3fold 5 repeats-> average predictions from each model trained
    #write wrapper for special regression that is binned into 4 categories...
    
    #sample weights
    
    #clean data remove special characters ( - + ) "
    #https://www.kaggle.com/triskelion/crowdflower-search-relevance/normalized-kaggle-distance
    #LDA visualisation: https://www.kaggle.com/solution/crowdflower-search-relevance/lda-visualization
    #LINEARsvm
    #http://streamhacker.com/2014/12/29/word2vec-nltk/
    # tune min_samples_leaf for RF
    # using regression for ensembling...
    #OK we do not use test set for training....???
    #test map with xgboost
    #remove models trained with test data
    #STOPWORDS = nltk.corpus.stopwords.words('english')
    #char ngrams
    #use bagging classifier
    #write wrapper for special regression that is binned into 4 categories...
    #abbreviations...
    #onehotencoding with countvectorizers
    #real majority voting!!!!
    #remove crappy ids : https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/14981/odd-search-result-in-train-csv
   
    t0 = time()
    
    print "numpy:",np.__version__
    print "pandas:",pd.__version__
    print "scipy:",sp.__version__
    
    seed = 123
    nsamples=-1#'shuffle'
    cleanse='load'
    useOnlyTrain=False
    concatTitleDesc=None#title+desription### quite bad...
    doBenchMark=False
    computeFeatures=True#'load'#True
    computeSim=True#True
    computeSynonyma=None##
    computeKaggleDistance=None#'load'##
    computeKaggleTopics=None#["notebook","computer","movie","clothes","media","shoe","kitchen","car","bike","toy","phone","food","sport"]
    computeWord2Vec=None#'load'
    addNoiseColumns=None
    vectorizeFirstOnly=False#True seems a bit better...?
    doSeparateTFID=['query','product_title']#['query','product_title','product_description']#
    doSVDseparate=261#10#261
    doSVDseparate_2nd=300#10#200
    doTFID=False
    doSVD=False
    
    standardize=True
    useAll=False#supervised vs. unsupervised
    concat=False#query+title      
    doKmeans=False
    
    #garbage=["<.*?>", "http", "www","img","border","style","px","margin","left", "right","font","solid","This translation tool is for your convenience only.*?Note: The accuracy and accessibility of the resulting translation is not guaranteed"]
    stop_words = text.ENGLISH_STOP_WORDS.union(corpus.stopwords.words('english'))
    
    #vectorizer = HashingVectorizer(stop_words=stop_words,ngram_range=(1,2),analyzer="char", non_negative=True, norm='l2', n_features=2**18)
    #vectorizer = TfidfVectorizer(min_df=5,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(2, 6), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2')#default
    vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='char',ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}',norm='l2')#default
    Xtrain, ytrain, Xtest,idx, sample_weight  = prepareDataset(seed=seed,nsamples=nsamples,vectorizer=vectorizer,cleanse=cleanse,useOnlyTrain=useOnlyTrain,doBenchMark=doBenchMark,computeWord2Vec=computeWord2Vec,computeFeatures=computeFeatures,computeSynonyma=computeSynonyma,computeKaggleDistance=computeKaggleDistance,computeKaggleTopics=computeKaggleTopics,addNoiseColumns=addNoiseColumns,concat=concat,doTFID=doTFID,doSeparateTFID=doSeparateTFID,vectorizeFirstOnly=vectorizeFirstOnly,stop_words=stop_words,computeSim=computeSim,doSVD=doSVD,doSVDseparate=doSVDseparate,doSVDseparate_2nd=doSVDseparate_2nd,standardize=standardize,useAll=useAll,concatTitleDesc=concatTitleDesc,doKmeans=doKmeans)
    sample_weight=None
    Xtrain.to_csv('./data/Xtrain.csv',index=False)
    pd.DataFrame(ytrain,columns=['class']).to_csv('./data/ytrain.csv',index=False)
    #model = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.001, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
    
    #model = Pipeline([('reducer', TruncatedSVD(n_components=400)), ('scaler', StandardScaler()), ('model', model)])
    #model = Pipeline([('scaler', StandardScaler()), ('model', SVC(C=10,gamma='auto') )])
    #model = SVC(C=10,gamma=0.001)
    #model = Pipeline([('scaler', StandardScaler()), ('model', LinearSVC(C=0.01))])
    #model = Pipeline([('reducer', MiniBatchKMeans(n_clusters=400,batch_size=400,n_init=3)), ('scaler', StandardScaler()), ('SVC', model)])
    #model = Pipeline([('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),('classification', LinearSVC())])
    #model = LinearSVC(C=0.01)
    
    
    #model =  RandomForestClassifier(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=1,criterion='gini', max_features=100)#0.627
    #model =  RandomForestClassifier(n_estimators=250,max_depth=None,min_samples_leaf=1,n_jobs=2,criterion='gini', max_features='auto')
    #model =  ExtraTreesClassifier(n_estimators=500,max_depth=None,min_samples_leaf=3,n_jobs=2,criterion='gini', max_features=150) #0.649
    #model = XgboostClassifier(n_estimators=500,learning_rate=0.2,max_depth=6,subsample=.5,n_jobs=1,objective='multi:softmax',eval_metric='error',booster='gbtree',silent=1)
    #model.fit(Xtrain,ytrain)
    #rfFeatureImportanccomputeSynonymae(model,Xtrain,Xtest,1)
    
    #model = OneVsRestClassifier(model,n_jobs=8)
    #model = OneVsOneClassifier(model,n_jobs=8)
    #model = SGDClassifier(alpha=.001, n_iter=200,penalty='l2',shuffle=True,loss='log')
    #model = SGDClassifier(alpha=0.0005, n_iter=50,shuffle=True,loss='log',penalty='l2',n_jobs=4)#opt  
    #model = SGDClassifier(alpha=0.0001, n_iter=50,shuffle=True,loss='log',penalty='l2',n_jobs=4)#opt simple processing
    #model = SGDClassifier(alpha=0.00014, n_iter=50,shuffle=True,loss='log',penalty='elasticnet',l1_ratio=0.99)
    #model = LogisticRegression(penalty='l2', tol=0.0001, C=1.0)#opt
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=50)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=1.0))])
    #model = Pipeline([('filter', SelectPercentile(chi2, percentile=70)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=1.0))])
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=15)), ('model', KNeighborsClassifier(n_neighbors=150))])
    #model = Pipeline([('filter', SelectPercentile(chi2, percentile=20)), ('model', MultinomialNB(alpha=0.1))])
    #model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None)#opt kaggle params
    #model = LogisticRegressionMod(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None)#opt kaggle params
    #model = RidgeClassifier(tol=1e-2, solver="lsqr",alpha=0.1)
    #model = KNeighborsClassifier(n_neighbors=5)
    #model = XgboostClassifier(n_estimators=200,learning_rate=0.001,max_depth=20,subsample=.5,n_jobs=1,objective='multi:softmax',eval_metric='merror',booster='gbtree',silent=1,eval_size=0.0)
    #model = XgboostClassifier(n_estimators=400,learning_rate=0.1,max_depth=4,subsample=.5,colsample_bytree=1.0,n_jobs=1,objective='multi:softmax',eval_metric='quadratic_weighted_kappa',booster='gbtree',silent=1,eval_size=0.3)
    #model.fit(Xtrain,ytrain)
    model = nnet_crowd
    
    #model.fit(Xtrain,ytrain)
    #model = MultinomialNB(alpha=0.001)
  
  
    #model = BaggingClassifier(base_estimator=model,n_estimators=3,n_jobs=1,verbose=2,random_state=None,max_samples=1.0,max_features=1.0,bootstrap=False)
    # Kappa Scorer 
    scoring_func = make_scorer(quadratic_weighted_kappa, greater_is_better = True)
    
    
    if isinstance(model,NeuralNet):
	  ytrain = ytrain - 1
    
    #labels = pd.read_csv('unique_queries.csv',index_col=0).values.flatten().astype(np.int32)
    #cv=StratifiedShuffleSplit(labels,8,test_size=0.3)
    cv=StratifiedShuffleSplit(ytrain,8,test_size=0.3)
    #labels = pd.read_csv('unique_titles.csv',index_col=0).values.flatten().astype(np.int32)%n_labels
    #print labels
    #print np.unique(labels)
    #cv= cross_validation.LeavePLabelOut(labels,p=2)
    #print "unique labels: %r cv sts: %d"%(np.unique(labels),len(cv))
    
    #cv =StratifiedKFold(ytrain,5,shuffle=True)
    #parameters = {'reducer__n_components': [200,300,400,500,600]}
    #parameters = {'reducer__n_clusters': [1000,1200,1500]}
    #parameters = {'SVC__C': [8,16,32],'SVC__gamma':[0.001,0.003,0.0008]}
    #parameters = {'C': [10],'gamma':[0.001]}
    #parameters = {'C': [0.5,0.1,0.05,0.01]}
    #parameters = {'alpha': [0.005,0.001,0.0025],'n_iter':[200],'penalty':['l2'],'loss':['log','perceptron']}
    #parameters = {'n_estimators':[500],'max_depth':[10],'learning_rate':[0.1,0.05,0.01],'subsample':[0.5]}
    #parameters = {'n_estimators':[250,500],'max_features':[100,150],'criterion':['gini'],'min_samples_leaf':[1,3]}
    #parameters = {'alpha':[0.001]}
    #parameters = {'dropout0_p':[0.25],'hidden1_num_units': [200],'dropout1_p':[0.25],'hidden2_num_units': [200],'dropout2_p':[0.5],'max_epochs':[50]}
    parameters = {'dropout0_p':[0.1,0.25],'hidden1_num_units': [400],'dropout1_p':[0.25,0.5],'hidden2_num_units': [400],'dropout2_p':[0.25,0.5],'max_epochs':[50,75,100]}
    
    model = makeGridSearch(model,Xtrain,ytrain,n_jobs=1,refit=False,cv=cv,scoring=scoring_func,parameters=parameters,random_iter=5)
    #model = buildModel(model,Xtrain,ytrain,cv=cv,scoring=scoring_func,n_jobs=4,trainFull=True,verbose=True)
    
    #model = buildClassificationModel(model,Xtrain,ytrain,sample_weight=sample_weight,class_names=['1','2','3','4'],trainFull=False,cv=cv)
    #makePredictions(model,Xtest,idx=idx,filename='submissions/sub03072015a.csv')
    print("Model building done in %fs" % (time() - t0))