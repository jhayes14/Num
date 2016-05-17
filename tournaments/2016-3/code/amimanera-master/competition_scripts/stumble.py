#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  Code scored 25th place in Kaggle competition, modified starter code from BSMan@Kaggle
"""

from qsprLib import *

import json

from nltk import word_tokenize,sent_tokenize
#from nltk.stem import SnowballStemmer # no english?
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
#from nltk.stem.wordnet import WordNetStemmer
#nltk.stem.porter.PorterStemmer(ignore_stopwords=False)
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import GermanStemmer
#http://nltk.googlecode.com/svn/trunk/doc/howto/collocations.html

from crawldata import *
from featureDesign import *
from FullModel import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#TODO build class containing classifiers + dataset
#TODO remove infrequent sparse features to new class
#TODO https://github.com/cbrew/Insults/blob/master/Insults/insults.py
#TODO look at wrong classified ones
#TODO analyze misclassifications
#TODO remove duplicate features boilerplate length & food stuff
#TODO http://nltk.org/book/ch05.html
#TODO http://scikit-learn.org/stable/auto_examples/plot_rfe_with_cross_validation.html#example-plot-rfe-with-cross-validation-py
#TODO using LDA with gensim: http://blog.kaggle.com/2012/07/17/getting-started-with-the-wordpress-competition/
#TODO recursive feature engineering
#TODO top:SDG_alpha0.000136463620667_L10.992081466188
#TODO bumping ->NO
#TODO calibration of AUC by reducing uncertain webpages to p=0.5 ->NO
#TODO transformation of variable log of length variables, standardize ->NO
#TODO calibration-> lof>x then  p=0.5+ - >NO
#TODO log transform ->NO...!!
#TODO use meta features.... -YES
#TODO dicretize continous data by cut and qcut to enlarge sparse matrix ...?
#TODO use new features to train xtra rf -YES
#TODO pickle to save model + datasets
#TODO mix features...https://github.com/tuzzeg/detect_insults/blob/master/README.md
#TODO https://class.coursera.org/nlp/lecture/index and http://nlp.stanford.edu/~wcmac/papers/20050421-smoothing-tutorial.pdf
#TODO analyse alt= data!!!
#TODO use lof as regular feature -> YES!!!
#TODO use separate model fpr foodstuff
#TODO SVD number of iterations
#TODO iterative selection of meta features...witihn xval loop!!!
#TODO check predictions...
#TODO scale ensemble!!!!!
#TODO anneal for best ensemble: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.anneal.html#scipy.optimize.anneal


class NLTKTokenizer(object):
    """
    http://scikit-learn.org/stable/modules/feature_extraction.html
    http://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
    http://nltk.org/api/nltk.tokenize.html
    """
    def __init__(self):
      #self.wnl = LancasterStemmer()
      self.wnl = PorterStemmer()#best so far
      #self.wnl = GermanStemmer()
      #self.wnl = EnglishStemmer(ignore_stopwords=True)
      #self.wnl = WordNetStemmer()
    def __call__(self, doc):
      words=[word_tokenize(t) for t in sent_tokenize(doc)]
      words=[item for sublist in words for item in sublist]
      if hasattr(self.wnl,'stem'):
	  words=[self.wnl.stem(t) for t in words]
      else:
	  words=[self.wnl.lemmatize(t) for t in words]
      return words
    
def dfinfo(X_all):
    print "##Basic data##\n",X_all
    print "##Details##\n",X_all.ix[:,0:2].describe()
    print "##Details##\n",X_all.ix[:,2:3].describe()
    print "##Details##\n",X_all.ix[:,3:7].describe()

def prepareDatasets(vecType='hV',useSVD=0,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=False,useGreedyFilter=False,char_ngram=5,loadTemp=False,usewordtagSmoothing=False,usetagwordSmoothing=False,usePosTagNew=False,useNLTKprob=False):
    """
    Load Data into pandas and preprocess features
    """
    print "loading dataset..."
    X = pd.read_csv('../stumbled_upon/data/train.tsv', sep="\t", na_values=['?'], index_col=1)
    X_test = pd.read_csv('../stumbled_upon/data/test.tsv', sep="\t", na_values=['?'], index_col=1)
    y = X['label']
    y = pd.np.array(y)
    X = X.drop(['label'], axis=1)
    # Combine test and train while we do our preprocessing
    X_all = pd.concat([X_test, X])
    print "Original shape:",X_all.shape
    
    if loadTemp:
	Xs = pd.read_csv('../stumbled_upon/data/Xtemp.csv', sep=",", index_col=0)
	Xs_test = pd.read_csv('../stumbled_upon/data/Xtemp_test.csv', sep=",", index_col=0)
	
	return (Xs,y,Xs_test,X_test.index,X.index)
    
    #vectorize data#
    #vectorizer = HashingVectorizer(ngram_range=(1,2), non_negative=True)
    if vecType=='hV':
	warnings.filterwarnings("ignore", category=UserWarning)
	print "Using hashing vectorizer..."
	#vectorizer = HashingVectorizer(stop_words='english',ngram_range=(1,2),analyzer="word", non_negative=True, norm='l2', n_features=2**19)
	vectorizer = HashingVectorizer(stop_words=None,ngram_range=(char_ngram,char_ngram),analyzer="char", non_negative=True, norm='l2', n_features=2**18)
    elif vecType=='tfidfV':
	print "Using tfidfV..."
	vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2),stop_words=None,max_features=None,binary=False,min_df=4,strip_accents='unicode',tokenizer=NLTKTokenizer())
	#vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2),stop_words=None,max_features=None,binary=True,min_df=5,strip_accents='unicode')
	#vectorizer = TfidfVectorizer(ngram_range=(1,1),max_features=2**14,sublinear_tf=True,min_df=3,tokenizer=NLTKTokenizer(),stop_words=None)
	#vectorizer = TfidfVectorizer(ngram_range=(1,1),max_features=2**14,sublinear_tf=True,min_df=2,stop_words=None)#fast
	#vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), sublinear_tf=True, norm=u'l2')#opt
    elif vecType=='tfidfV_small':
	vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=2**12,sublinear_tf=True,min_df=4,stop_words=None)#fast
    elif vecType=='tfidfV_large':
	vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2),stop_words=None,max_features=None,binary=False,min_df=2,strip_accents='unicode',tokenizer=NLTKTokenizer())
	#vectorizer = TfidfVectorizer(min_df=2,  max_features=None, strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), sublinear_tf=True, norm=u'l2')#opt
    elif vecType=='test':
       print "Test mode..."
       #vectorizer = CountVectorizer(ngram_range=(1,2),analyzer='word',max_features=2**14,min_df=3,tokenizer=NLTKTokenizer(),stop_words=None)
       vectorizer = HashingVectorizer(binary=False,stop_words=None,ngram_range=(char_ngram,char_ngram),analyzer="char", non_negative=True, norm='l2', n_features=2**14)
       #vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 1), sublinear_tf=True, norm=u'l2')
       #vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2),stop_words=None,max_features=None,binary=False,min_df=2,strip_accents='unicode',tokenizer=NLTKTokenizer())
    else:
	print "Using count vectorizer..."
	#vectorizer = CountVectorizer(ngram_range=(1,2),analyzer='word',max_features=2**18)
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char_wb",ngram_range=(4,4),max_features=2**14,stop_words='english')#AUC = 0.781
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(4,4),max_features=2**14,stop_words='english')#AUC= 0.786
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(4,4),max_features=2**18,stop_words='english')#AUC= 0.798
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(5,5),max_features=2**14,stop_words='english')#slow and low score 0.786
	vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(char_ngram,char_ngram),max_features=2**18,stop_words=None)#AUC=  0.815 1400s
	#vectorizer = CountVectorizer(lowercase=True,analyzer="char",ngram_range=(5,5),max_features=2**18,stop_words='english')#AUC= 0.813  1400s
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(4,5),max_features=2**18,stop_words='english')#AUC=   aufgehängt memory?
	#vectorizer = CountVectorizer(lowercase=False,analyzer="char",ngram_range=(5,5),max_features=2**16,stop_words=None)#AUC= 0.806 682s
    
    #transform data using json
    if useJson:
	print "Xtracting data using json..."
	#take only boilerplate data
	X_all['boilerplate'] = X_all['boilerplate'].apply(json.loads)
	
	#print X_all['boilerplate']
	#print X_all['boilerplate'][2]
	# Initialize the data as a unicode string
	X_all['body'] = u'empty'
	extractBody = lambda x: x['body'] if x.has_key('body') and x['body'] is not None else u'empty'
	X_all['body'] = X_all['boilerplate'].map(extractBody)
	
	X_all['title'] = u'empty'
	extractBody = lambda x: x['title'] if x.has_key('title') and x['title'] is not None else u'empty'
	X_all['title'] = X_all['boilerplate'].map(extractBody)
	
	X_all['url2'] = u'empty'
	extractBody = lambda x: x['url'] if x.has_key('url') and x['url'] is not None else u'empty'
	X_all['url2'] = X_all['boilerplate'].map(extractBody)
	
	X_all['body'] = X_all.body+u' '+X_all.url2
	X_all['body'] = X_all.body+u' '+X_all.title
	#print X_all['body'].head(30).to_string()
	#print X_all['body'].tail(30).to_string()		
	body_counts = vectorizer.fit_transform(X_all['body'])
	
	#body_counts = body_counts.tocsr()
	print "body,title+url, dim:",body_counts.shape
	print "density:",density(body_counts)		

    #simple transform
    else:
        print "Creating dataset by simple method..."
        body_counts=list(X_all['boilerplate'])
	body_counts = vectorizer.fit_transform(body_counts)
	print "Final dim:",body_counts.shape	
    #feature_names = None
    #if hasattr(vectorizer, 'get_feature_names'):
    	#feature_names = np.asarray(vectorizer.get_feature_names())
    #X & X_test are converted to sparse matrix
    
    #bringt seltsamerweise nichts
    #X_alcat=pd.DataFrame(X_all['alchemy_category'])
    #X_alcat=X_alcat.fillna('NA')
    #X_alcat = one_hot_encoder(X_alcat, ['alchemy_category'], replace=True)
    #print X_alcat
    #X_alcat=sparse.csr_matrix(pd.np.array(X_alcat))
    #body_counts = sparse.hstack((body_counts,X_alcat),format="csr")

    if useSVD>1:
	if useAddFeatures==True:
	    X_raw=crawlRawData(X_all)
	    X_all=pd.concat([X_all,X_raw], axis=1)
	    X_all = featureEngineering(X_all)
	
	print "Actual shape:",X_all.shape
	#SVD of text data (LSA)
	print "SVD of sparse data with n=",useSVD
	tsvd=TruncatedSVD(n_components=useSVD, algorithm='randomized', n_iterations=5, tol=0.0)
	X_svd=tsvd.fit_transform(body_counts)
	X_svd=pd.DataFrame(np.asarray(X_svd),index=X_all.index)
	#char_ngrams
	if useHTMLtag:
	    #char ngrams
	    X_raw=crawlHTML(X_all)
	    #char_vectorizer=CountVectorizer(lowercase=False,analyzer="char",ngram_range=(5,5),max_features=2**18,stop_words=None)
	    char_vectorizer=CountVectorizer(lowercase=True,analyzer="char",ngram_range=(5,5),max_features=2**14,stop_words=None)
	    char_ngrams = char_vectorizer.fit_transform(X_raw['htmltag'])
	    #char_ngrams = char_vectorizer.fit_transform(X_all['body'])
	    print "char ngrams, dim:",char_ngrams.shape
	    print "density:",density(char_ngrams)
	    useSVD2=10
	    print "SVD of char ngrams data with n=",useSVD2
	    tsvd=TruncatedSVD(n_components=useSVD2, algorithm='randomized', n_iterations=5, tol=0.0)
	    X_char=tsvd.fit_transform(char_ngrams)
	    X_char=pd.DataFrame(np.asarray(X_char),index=X_all.index,columns=["char"+str(x) for x in xrange(useSVD2)])
	    print "X_char",X_char
	    X_svd=pd.concat([X_svd,X_char], axis=1)
	
	if usePosTag:
	    #posTagging(X_all)
	    X_pos=pd.read_csv('../stumbled_upon/data/postagged.csv', sep=",", na_values=['?'], index_col=0)
	    X_svd=pd.concat([X_svd,X_pos], axis=1)
	    #new postag
	if usePosTagNew:
	    X_pos=pd.read_csv('../stumbled_upon/data/postagfeats.csv', sep=",", na_values=['?'], index_col=0)
	    X_svd=pd.concat([X_svd,X_pos], axis=1)
	
	#print "##X_svd##\n",X_svd
	X_all= X_all.drop(['boilerplate','url'], axis=1)
	X_all= X_all.drop(['hasDomainLink','framebased','news_front_page','embed_ratio'], axis=1)
	if useJson: 
	    X_rest= X_all.drop(['body','url2','title','alchemy_category'], axis=1)	    
	else:
	    X_rest= X_all.drop(['alchemy_category'], axis=1)
	X_rest = X_rest.astype(float)
	X_rest=X_rest.fillna(X_rest.median())	
	X_rest.corr().to_csv("corr.csv")
	X_svd=pd.concat([X_rest,X_svd], axis=1)
	#print "##X_svd,int##\n",X_svd
	#add alchemy category again, but now one hot encode, bringt nichts...
	if useAlcat:
	    X_alcat=pd.DataFrame(X_all['alchemy_category'])
	    X_alcat=X_alcat.fillna('unknown')
	    X_alcat = one_hot_encoder(X_alcat, ['alchemy_category'], replace=True)
	    X_alcat = pd.DataFrame(X_alcat)
	    X_svd=pd.concat([X_svd,X_alcat], axis=1)
	   
	
	if useGreedyFilter:
	    print X_svd
	    #print X_svd.columns
	    #X_svd=X_svd.loc[:,[1,4,3,8,5,'linkwordscore',6,'char2',9,'url_contains_foodstuff',22,26,'MOD',33,'alchemy_category_score',24,45,'spelling_errors_ratio',43]]#Rgreedy
	    #X_svd=X_svd.loc[:,[1,4,3,8,5,'linkwordscore',6,'char2',9,'url_contains_foodstuff',22,26,'MOD',33,'alchemy_category_score',24,45,'spelling_errors_ratio',43,'frameTagRatio',19,21,25,0,'url_length',48,'TO','char5','url_contains_news','compression_ratio',37,'VD','twitter_ratio',49,'is_news','url_contains_sweetstuff',42,17,'url_contains_health',20,'char4',16,'DET',23,'commonlinkratio_2',41,'image_ratio',7,'wwwfacebook_ratio','char0']]
	    X_svd=X_svd.loc[:,[1, 2, 4, u'url_contains_foodstuff', 9, 0, 8, u'CNJ', u'url_contains_recipe', 33, 6, u'non_markup_alphanum_characters', 10, u'body_length', 15, 5, 3, u'char2', 12, 11, 14, 21, 31, u'frameTagRatio', 7, 25, u'N', 22, 17, 16, 23, 19, 47, 18, u'linkwordscore', 29, 46, 30, u'V', 39, 32]]#rf feature importance sklearn
	    #X_svd=X_svd.loc[:,[1, 4, 9, 8, 0, u'url_contains_foodstuff', 2, 58, 67, 75, 71, u'CNJ', 44, 21, u'non_markup_alphanum_characters', 15, 42, u'linkwordscore', u'frameTagRatio', 36, u'logn_newline', 65, 47, 29, 33, 64, u'body_length', u'DET', 73, 56, 12, u'P', 14, 6, u'char0', 97, 37, 52, 83, 79, 17, u'avglinksize', u'char8', 22, 39, u'char4', u'wwwfacebook_ratio', u'url_length', u'ADJ', u'char1', 85, 30, 72, 62, 49, 28, 11, 59, 89, u'n_comment', 78, 55, 53, u'MOD', u'compression_ratio', 54, u'spelling_errors_ratio', u'commonlinkratio_1']]#GBM feature selection
	    #X_svd=X_svd.loc[:,[1, 4, 9, 8, 0, 2, u'url_contains_foodstuff', 58, 67, 75, u'CNJ', u'linkwordscore', 21, 64, 15, u'non_markup_alphanum_characters', u'frameTagRatio', 44, 47, 42, u'P', 29, 36, 33, 73, u'DET', 97, 12, 56, 71, 52, 65, 59, 37, u'logn_newline', 6, 85, u'char0', 22, 83, u'url_length', u'body_length', 14, 17, 30, u'avglinksize', 62, u'compression_ratio', 39, u'ADJ', u'char1', 53, 78, 49, 11, 54, 79, 89, u'char4', u'char8']]#GBM feature selection
	    #X_svd=X_svd.loc[:,[1,4,3,8,5,u'linkwordscore']]
	    print X_svd
	
	
	print "##X_svd,final##\n",X_svd
	#X_rest=X_svd
	print "Dim: X_svd:",X_svd.shape    
	X_svd_train = X_svd[len(X_test.index):]
	X_svd_test = X_svd[:len(X_test.index)]
	return(X_svd_train,y,X_svd_test,X_test.index,X.index)
    else:
	if usePosTag:
	    #create new body with postag features!
	    #postagFeatures(X_all)
	    X_pos=pd.read_csv('../stumbled_upon/data/postagfeats.csv', sep=",", na_values=['?'], index_col=0)
	    #X_pos=pd.read_csv('../stumbled_upon/data/postagfeats_simple.csv', sep=",", na_values=['?'], index_col=0)
	    
	    #vectorize data
	    #vectorizer = CountVectorizer(ngram_range=(1,1),analyzer='word',max_features=2**14)
	    #vectorizer = HashingVectorizer(ngram_range=(1,1),analyzer='word',token_pattern=r'\w{1,}')
	    vectorizer =TfidfVectorizer(min_df=4,  max_features=2**18, strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 7), sublinear_tf=True, norm=u'l2')
	    poscounts=vectorizer.fit_transform(X_pos['postagfeats'])  
	    #print vectorizer.get_feature_names()
	    print poscounts.shape
	    #print body_counts.shape
	    body_counts = sparse.hstack((body_counts,poscounts),format="csr")
	    #body_counts=poscounts
	 
	if useNLTKprob:
	    lidstoneProbDist(X_all)
	
	if usewordtagSmoothing or usetagwordSmoothing:
	    #create tag-words and word-tag
	    #postagSmoothing(X_all)
	    X_smoothed=pd.read_csv('../stumbled_upon/data/postagsmoothed2.csv', sep=",", na_values=['?'], index_col=0,encoding='utf-8')
	    print X_smoothed
	    
	    #word-tags
	    #vectorizer = CountVectorizer(ngram_range=(1,1),analyzer='word',max_features=2**10,token_pattern=r'\w{1,}')
	    #vectorizer = HashingVectorizer(ngram_range=(1,1),analyzer='word',max_features=2**18,token_pattern=r'\w{1,}')
	    vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), sublinear_tf=True, norm=u'l2')
	    if usewordtagSmoothing:
		print "Using word-tag smoothing..." 
		col='wordtag'
		wordtagcounts=vectorizer.fit_transform(X_smoothed[col])
		print wordtagcounts.shape
		body_counts=wordtagcounts
		#body_counts = sparse.hstack((body_counts,wordtagcounts),format="csr")
		
	    if usetagwordSmoothing:
		print "Using tag-word smoothing..."
		col='tagword'
		tagwordcounts=vectorizer.fit_transform(X_smoothed[col])
		print tagwordcounts.shape
		body_counts=tagwordcounts
		#body_counts = sparse.hstack((body_counts,tagwordcounts),format="csr")	
		
	    #print "Feature names:",vectorizer.get_feature_names()
	    print type(body_counts)
	    print body_counts.shape
	    
	    print "density:",density(body_counts)
            
	Xs = body_counts[len(X_test.index):]
	Xs_test = body_counts[:len(X_test.index)]
	#conversion to array necessary to work with integer indexing, .iloc does not work with this version
	return (Xs,y,Xs_test,X_test.index,X.index)
	
def splitModel(lmodel,lXs,lXs_test,ly):
    """
    separate model for foodstuff, should be done for sparse matrix stuff! should be done in XVAL loop
    #http://stackoverflow.com/questions/12213818/splitting-a-sparse-matrix-into-two
    #at the ensemble level?
    """  
    ly=pd.DataFrame(ly)
    ly.index=lXs.index
    lXall=pd.concat([lXs,ly],axis=1)
    
    #FOOD DATA
    lXall_food = pd.DataFrame(lXall[lXall['url_contains_recipe']> 0.5])
    ly = lXall_food.ix[:,-1]
    lXall_food=lXall_food.ix[:,:-1] 
    print "Food data set: ",lXall_food.shape   
    parameters = {'n_estimators':[200,500], 'max_features':['auto']}#rf
    clf_opt = grid_search.GridSearchCV(lmodel, parameters,cv=8,scoring='roc_auc',n_jobs=4,verbose=1)
    clf_opt.fit(lXall_food,ly)
    for params, mean_score, scores in clf_opt.grid_scores_:
        print("%0.3f (+/- %0.3f) for %r"
              % (mean_score.mean(), scores.std(), params))
    
    #NON-FOOD DATA
    lXall_rest = pd.DataFrame(lXall[lXall['url_contains_foodstuff']<= 0.5])
    ly = lXall_rest.ix[:,-1]
    lXall_rest=lXall_rest.ix[:,:-1]
    print "Non food data: ",lXall_rest.shape
    clf_opt = grid_search.GridSearchCV(lmodel, parameters,cv=8,scoring='roc_auc',n_jobs=4,verbose=1)
    clf_opt.fit(lXall_rest,ly)
    for params, mean_score, scores in clf_opt.grid_scores_:
        print("%0.3f (+/- %0.3f) for %r"
              % (mean_score.mean(), scores.std(), params))
    
if __name__=="__main__":
    """   
    MAIN PART
    """ 
    # Set a seed for consistant results
    t0 = time()
    np.random.seed(123)
    print "numpy:",np.__version__
    print "pandas:",pd.__version__
    #print pd.util.terminal.get_terminal_size()
    pd.set_printoptions(max_rows=300, max_columns=8)
    print "scipy:",sp.__version__
    print "nltk:",nltk.__version__
    #variables
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('hV',useSVD=500,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=True,useGreedyFilter=False)#opt SVD=50
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV_small',useSVD=50,useJson=True,useHTMLtag=False,useAddFeatures=True,usePosTag=False,useAlcat=False,useGreedyFilter=False)
    #Xs=pd.DataFrame(Xs.todense())
    #Xs_test=pd.DataFrame(Xs_test.todense())
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('hV',useSVD=10,useJson=False,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,char_ngram=1,loadTemp=True)
    (Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=1000,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=True,useGreedyFilter=False)#opt SVD=50
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=100,useJson=True,useHTMLtag=False,useAddFeatures=False,usePosTag=False,useAlcat=False,useGreedyFilter=False,loadTemp=True)#opt SVD=50
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV',useSVD=2,useJson=True,useHTMLtag=True,useAddFeatures=True,usePosTag=True,useAlcat=True,useGreedyFilter=False)#
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('test',useSVD=0,useJson=True,usePosTag=False,usewordtagSmoothing=False,usetagwordSmoothing=False,useNLTKprob=True)
    #(Xs,y,Xs_test,test_indices,train_indices) = prepareDatasets('tfidfV_large',useSVD=0,useJson=True)
    #Xs.to_csv("../stumbled_upon/data/Xens.csv")
    #Xs_test.to_csv("../stumbled_upon/data/Xens_test.csv")
    #Xs.to_csv("../stumbled_upon/data/Xlarge.csv")
    #Xs_test.to_csv("../stumbled_upon/data/Xlarge_test.csv")

    #(Xs,y,Xs_test,test_indices) = prepareSimpleData()
    print "Dim X (training):",Xs.shape
    print "Type X:",type(Xs)
    print "Dim X (test):",Xs_test.shape
    # Fit a model and predict
    #model = SGDClassifier(alpha=.0001, n_iter=50,penalty='elasticnet',l1_ratio=0.2,shuffle=True,loss='log')
    #model = SGDClassifier(alpha=0.0005, n_iter=50,shuffle=True,loss='log',penalty='l2',n_jobs=4)#opt  
    #model = SGDClassifier(alpha=0.0001, n_iter=50,shuffle=True,loss='log',penalty='l2',n_jobs=4)#opt simple processing
    #model = SGDClassifier(alpha=0.00014, n_iter=50,shuffle=True,loss='log',penalty='elasticnet',l1_ratio=0.99)
    model = LogisticRegression(penalty='l2', tol=0.0001, C=1.0)#opt
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=50)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=1.0))])
    #model = Pipeline([('filter', SelectPercentile(chi2, percentile=70)), ('model', LogisticRegression(penalty='l2', tol=0.0001, C=1.0))])
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=15)), ('model', KNeighborsClassifier(n_neighbors=150))])
    #model = Pipeline([('filter', SelectPercentile(chi2, percentile=20)), ('model', MultinomialNB(alpha=0.1))])
    #model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None)#opt kaggle params
    #model = LogisticRegressionMod(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None)#opt kaggle params
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=100)), ('model', LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=10.0, class_weight=None))])
    #model = AdaBoostClassifier(base_estimator=LogisticRegressionMod(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True,intercept_scaling=1.0),learning_rate=0.1,n_estimators=50,algorithm="SAMME.R")
    #model = KNeighborsClassifier(n_neighbors=10)
    #model=SVC(C=0.3,kernel='linear',probability=True)
    #model=LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0)#no proba
    #model = SVC(C=1, cache_size=200, class_weight='auto', gamma=0.0, kernel='linear', probability=True, shrinking=True,tol=0.001, verbose=False)
    #model=   RandomForestClassifier(n_estimators=200,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features='auto',oob_score=False)
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=80)), ('model', AdaBoostClassifier(n_estimators=100,learning_rate=0.1))])
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=50)), ('model', BernoulliNB(alpha=0.1))])#opt sparse 0.849
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=50)), ('model', RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_leaf=10,n_jobs=1,criterion='entropy', max_features='auto',oob_score=False))])
    #opt greedy approach
    #model = AdaBoostClassifier(n_estimators=500,learning_rate=0.1)

    #model = ExtraTreesClassifier(n_estimators=100,max_depth=None,min_samples_leaf=10,n_jobs=4,criterion='entropy', max_features=20,oob_score=False)#opt
    #model = AdaBoostClassifier(n_estimators=100,learning_rate=0.1)
    #model = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=500, subsample=0.5, min_samples_split=6, min_samples_leaf=10, max_depth=5, init=None, random_state=123,verbose=False)#opt 0.883
    #model = SVC(C=1, cache_size=200, class_weight='auto', gamma=0.0, kernel='rbf', probability=True, shrinking=True,tol=0.001, verbose=False)  
    #modelEvaluation(model,Xs,y)
    #model=pyGridSearch(model,Xs,y)
    #splitModel(model,Xs,Xs_test,y)
    #(gclassifiers,gblender,oob_avg)=ensembleBuilding(Xs,y)
    #ensemblePredictions(gclassifiers,gblender,Xs,y,Xs_test,test_indices,train_indices,oob_avg,'../stumbled_upon/data/lgblend.csv')
    #fit final model
    #(Xs,Xs_test)=scaleData(Xs,Xs_test,['body_length','linkwordscore','frameTagRatio','non_markup_alphanum_characters'])
    #Xs.hist()
    #print model
    model = buildModel(model,Xs,y) 
    #print model.estimator_errors_
    #print model.estimator_weights_
    #showMisclass(Xs,Xs_test,y)
    #(Xs,Xs_test,y)=filterClassNoise(model,Xs,Xs_test,y)
    #model = buildModel(model,Xs,y) 
    #(Xs,Xs_test)=iterativeFeatureSelection(model,Xs,Xs_test,y,1,1)
    #Xs.to_csv("../stumbled_upon/data/Xtemp.csv")
    #Xs_test.to_csv("../stumbled_upon/data/Xtemp_test.csv")
    #model = buildModel(model,Xs,y) 
    #lofFilter(y)
    #(Xs,Xs_test) = group_sparse(Xs,Xs_test)
    #print "Dim X (after grouping):",Xs.shape
    #makePredictions(model,Xs_test,test_indices,'../stumbled_upon/submissions/sub0410b.csv')	            
    print("Model building done in %fs" % (time() - t0))
    plt.show()
