#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  crawl data
"""

from qsprLib import *
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
import re
import difflib
from scipy.spatial.distance import cdist

import itertools
import math

import gensim
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk import Text

from kaggle_distance import *
import pickle

#TODO: right after tfidf, input 2 sparse matrics: def computeSimilarityFeatures(Xs_all,Xs_all_new)
#and for dense def computeSimilarityFeatures(Xall,Xall_new,nsplit)
#http://stackoverflow.com/questions/16597265/appending-to-an-empty-data-frame-in-pandas

 
def computeSimilarityFeatures(Xall,columns=['query','product_title'],verbose=False,useOnlyTrain=False,startidx=0,stop_words=None,doSVD=261):
    print "Compute scipy similarity..."
    vectorizer = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',ngram_range=(1, 5), use_idf=True,smooth_idf=True,sublinear_tf=True,stop_words = stop_words,token_pattern=r'\w{1,}')
    if useOnlyTrain:
      print "Using train only for TFIDF..."
      Xtrain = Xall[startidx:]
      Xs1 = vectorizer.fit(Xtrain[columns[0]])
    else:
      Xs1 = vectorizer.fit(Xall[columns[0]])
    
    Xs1 = vectorizer.transform(Xall[columns[0]])
    Xs2 = vectorizer.transform(Xall[columns[1]])
    #print "Xs1",Xs1.shape
    #print "Xs2",Xs2.shape
    #print Xall['query'].iloc[:5]
    #print Xall['product_title'].iloc[:5]
    sparse=True
    if doSVD is not None:
	print "Similiarity with SVD, n_components:",doSVD
	reducer=TruncatedSVD(n_components=doSVD, algorithm='randomized', n_iter=5, tol=0.0)
	Xs1 = reducer.fit_transform(Xs1)
	Xs2 = reducer.transform(Xs2)
	sparse=False

    sim = computeScipySimilarity(Xs1,Xs2,sparse=sparse)
    return sim

def computeScipySimilarity(Xs1,Xs2,sparse=False):
    if sparse:
	Xs1 = np.asarray(Xs1.todense())
	Xs2 = np.asarray(Xs2.todense())
    Xall_new = np.zeros((Xs1.shape[0],2))
    for i,(a,b) in enumerate(zip(Xs1,Xs2)):
	a = a.reshape(-1,a.shape[0])
	b = b.reshape(-1,b.shape[0])
	#print a.shape
	#print type(a)
	dist = cdist(a,b,'cosine')
	#print dist
	#print type(dist)
	
	Xall_new[i,0] = dist	
	
	#dist = cdist(a,b,'minkowski')
	#Xall_new[i,3] = dist
	dist = cdist(a,b,'cityblock')
	Xall_new[i,1] = dist
	#dist = cdist(a,b,'hamming')
	#Xall_new[i,2] = dist
	#dist = cdist(a,b,'euclidean')
	#Xall_new[i,3] = dist
	#dist = cdist(a,b,'correlation')
	#Xall_new[i,5] = dist
	#dist = cdist(a,b,'jaccard')
	#Xall_new[i,3] = dist
	
    Xall_new = pd.DataFrame(Xall_new,columns=['cosine','cityblock'])
    print "NA:",Xall_new.isnull().values.sum()
    Xall_new = Xall_new.fillna(0.0)
    print "NA:",Xall_new.isnull().values.sum()
    print Xall_new.corr(method='spearman')
    return Xall_new


def getSynonyms(word,stemmer):
    #synonyms=[word]
    try:
      synonyms = [l.lemma_names() for l in wn.synsets(word)]
    except:
      pass 
    synonyms.append([word])
    synonyms = list(itertools.chain(*synonyms))
    synonyms = [stemmer.stem(l.lower()) for l in synonyms]
    synonyms = set(synonyms)
    return(synonyms)

def makeQuerySynonyms(Xall,construct_map=False):    
    query_map={}
    if construct_map:
      print "Creating synonyma for query..."
      model = gensim.models.Word2Vec.load_word2vec_format('/home/loschen/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)      
      X_temp = Xall.drop_duplicates('query')
      print X_temp.describe()
    
      for i in range(X_temp.shape[0]):
	  query = X_temp["query"].iloc[i].lower()
	  qsynonyms = query.split()
	  for word in query.split():
	      #print "word:",word
	      try:
		s = model.most_similar(word, topn=3)	      		
		qlist = []
		for item,sim in s:
		  if sim>0.6:
		    qlist.append(item.lower())
		  
		#print "word: %s synonyma: %r"%(word,qlist)
		qsynonyms = qsynonyms+qlist
	      except:
		pass
	  
	  #print qsynonyms
	  qsynonyms = (" ").join(z.replace("_"," ") for z in qsynonyms)
	  #print qsynonyms
	  #Xall["query"].iloc[i]=qsynonyms
	  query_map[query]=qsynonyms
	  #raw_input()
	  if i%20==0:
	    print "i:",i
    
      with open("w2v_querymap.pkl", "w") as f: pickle.dump(query_map, f) 
    
    with open("w2v_querymap.pkl", "r") as f: query_map = pickle.load(f)


    print "Mapping synonyma to query..."
    for i in range(Xall.shape[0]):
	query = Xall["query"].iloc[i].lower()
	Xall["query"].iloc[i]=query_map[query]
	if i%5000==0:
	  print "i:",i

    print Xall['query'].iloc[:10]
    return Xall


def information_entropy(text):
    log2=lambda x:math.log(x)/math.log(2)
    exr={}
    infoc=0
    for each in text:
        try:
            exr[each]+=1
        except:
            exr[each]=1
    textlen=len(text)
    for k,v in exr.items():
        freq  =  1.0*v/textlen
        infoc+=freq*log2(freq)
    infoc*=-1
    return infoc

#query id via label_encoder
#max similarity with difflib
#use kaggle distance??
#closed distance 
#
#text.similar('woman')
def additionalFeatures(Xall,verbose=False,dropList=['bestmatch']):
    #dropList=['bestmatch','S_title','S_query','checksynonyma']
    print "Computing additional features..."
    #text = Text(word.lower() for word in brown.words())
    stemmer = PorterStemmer()
    Xall_new = np.zeros((Xall.shape[0],13))
    for i in range(Xall.shape[0]):
	query = Xall["query"].iloc[i].lower()
	title = Xall["product_title"].iloc[i].lower()
	desc = Xall["product_description"].iloc[i].lower()
	
	#here we should get similars...
	similar_words = [getSynonyms(q,stemmer) for q in query.split()]
	similar_words = set(itertools.chain(*similar_words))
	
	query=re.sub("[^a-zA-Z0-9]"," ", query)
	query= (" ").join([stemmer.stem(z) for z in query.split()])
        
        title=re.sub("[^a-zA-Z0-9]"," ", title)
        title= (" ").join([stemmer.stem(z) for z in title.split()])
        
        desc=re.sub("[^a-zA-Z0-9]"," ", desc)
        desc= (" ").join([stemmer.stem(z) for z in desc.split()])
        
        nquery = len(query.split())
	ntitle = len(title.split())
	ndesc = len(desc.split())
	
	Xall_new[i,0] = nquery
	Xall_new[i,1] = ntitle
	Xall_new[i,2] = nquery / float(ntitle)
	Xall_new[i,3] = ndesc+1
	Xall_new[i,4] = nquery / float(ndesc+1)
	
	s = difflib.SequenceMatcher(None,a=query,b=title).ratio()
	
	Xall_new[i,5] = s

	nmatches = 0
	avgsim = 0.0
	lastsim = 0.0
	firstsim = 0.0
	checksynonyma = 0.0
	
	for qword in query.split():
	    if qword in title:
		nmatches+=1
		avgsim = avgsim + 1.0
		if qword == query.split()[-1]:
		  lastsim+=1
		if qword == query.split()[0]:
		  firstsim+=1
		
	    else:
	      bestvalue=0.0
	      for tword in title.split():
		s = difflib.SequenceMatcher(None,a=qword,b=tword).ratio()
		if s>bestvalue:
		    bestvalue=s
	      avgsim = avgsim + bestvalue
	      if qword == query.split()[-1]:
		  lastsim = lastsim + bestvalue
	      if qword == query.split()[0]:
		  firstsim = firstsim + bestvalue
	
	    #check similar
	    #print "qword:",qword
	    
	    #if similar_words is not None:
	      for simword in similar_words:
		  if simword in title:
		      checksynonyma+=1		  
		
	Xall_new[i,6] = nmatches / float(nquery)	
	Xall_new[i,7] = avgsim / float(nquery)	
	Xall_new[i,8] = information_entropy(query)
	Xall_new[i,9] = information_entropy(title)
	Xall_new[i,10] = lastsim
	Xall_new[i,11] = firstsim
	Xall_new[i,12] = checksynonyma / float(nquery)
	
	if i%5000==0:
	  print "i:",i
	
	if verbose:
	  print query
	  print nquery
	  print title
	  print ntitle
	  print "ratio:",Xall_new[i,2]
	  print "difflib ratio:",s
	  print "matches:",nmatches
	  raw_input()
	
    Xall_new = pd.DataFrame(Xall_new,columns=['query_length','title_length','query_title_ratio','desc_length','query_desc_ratio','difflibratio','bestmatch','averagematch','S_query','S_title','last_sim','first_sim','checksynonyma',]) 
    Xall_new = Xall_new.drop(dropList, axis=1)
    print Xall_new.corr(method='spearman')
    return Xall_new	

def cleanse_data(Xall):
    print "Cleansing the data..."
    with open("query_map.pkl", "r") as f: query_map = pickle.load(f)#key query value corrected value
  
    ablist=[]
    ablist.append((['ps','p.s.','play station','ps2','ps3','ps4'],'playstation'))
    ablist.append((['ny','n.y.'],'new york'))
    ablist.append((['tb','tera byte'],'gigabyte'))
    ablist.append((['gb','giga byte'],'gigabyte'))
    ablist.append((['t.v.','tv'],'television'))
    ablist.append((['mb','mega byte'],'megabyte'))
    ablist.append((['d.r.','dr'],'doctor'))
    ablist.append((['phillips'],'philips'))
    
    for i in range(Xall.shape[0]):
	query = Xall["query"].iloc[i].lower()
	
	#correct typos
	if query in query_map.keys():
	    query = query_map[query]
	
	#correct abbreviations query
	new_query =[]
	for qword in query.split():
	  for ab,full in ablist:
	    if qword in ab:
	      qword = full
	  new_query.append(qword)
	
	new_query = (" ").join(new_query)
	Xall["query"].iloc[i] = new_query
	
	title = Xall["product_title"].iloc[i].lower()
	
	#correct abbreviations title
	new_title=[]
	for qword in title.split():
	  for ab,full in ablist:
	    if qword in ab:
	      qword = full
	  new_title.append(qword)
	new_title = (" ").join(new_title)
	Xall["product_title"].iloc[i] = new_title
    
    
	if i%5000==0:
	      print "i:",i
    
    print "Finished"
    return Xall
    
    
#make top 5 most similar in query and check again...
def genWord2VecFeatures(Xall,verbose=True,dropList=[]):
    print "Compute word2vec features..."
    #print Xall['query'].tolist()
    #print brown.sents()
    #b = gensim.models.Word2Vec(brown.sents())
    model = gensim.models.Word2Vec.load_word2vec_format('/home/loschen/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
    Xall_new = np.zeros((Xall.shape[0],5))
    for i in range(Xall.shape[0]):
	query = Xall["query"].iloc[i].lower()
	title = Xall["product_title"].iloc[i].lower()
	
	query=re.sub("[^a-zA-Z0-9]"," ", query)
        nquery = len(query.split())
        
        title=re.sub("[^a-zA-Z0-9]"," ", title)
        ntitle = len(title.split())
        
	bestsim = 0.0
	lastsim = 0.0
	firstsim = 0.0
        avgsim = 0.0
        
        #print "Query:",query
        #print "Title:",title
        for qword in query.split():
	      if qword in title:
		bestsim = bestsim + 1.0
		avgsim = avgsim +1.0
		if qword == query.split()[-1]:
		  lastsim+=1
		if qword == query.split()[0]:
		  firstsim+=1
	      else:
		bestvalue=0.0
		for tword in title.split():
		  try:
		    s = model.similarity(qword,tword)
		    #print "query: %s title: %s  sim: %4.2f"%(qword,tword,s)
		    #print model.most_similar(qword, topn=5)
		    #print model.most_similar(tword, topn=5)
		  except:		    
		    s = 0.0
		  avgsim = avgsim + s
		  if s>bestvalue:
		      bestvalue=s
		  
		bestsim = bestsim + bestvalue
		#print "bestvalue: %4.2f avg: %4.2f"%(bestvalue,avgsim)
		
		if qword == query.split()[-1]:
		    lastsim = bestvalue
		if qword == query.split()[0]:
		    firstsim = bestvalue
		    
	if i%5000==0:
	      print "i:",i
    
	Xall_new[i,0] = bestsim / float(nquery)	
	Xall_new[i,1] = lastsim
	Xall_new[i,2] = firstsim
	Xall_new[i,3] = avgsim / float(ntitle)
	Xall_new[i,4] = avgsim 
	
	#raw_input()
    Xall_new = pd.DataFrame(Xall_new,columns=['w2v_bestsim','w2v_lastsim','w2v_firstsim','w2v_avgsim','w2v_totalsim'])
    Xall_new = Xall_new.drop(dropList, axis=1)
    print Xall_new.corr(method='spearman')
    return Xall_new
    
def createKaggleDist(Xall,general_topics=["notebook","computer","movie","clothes","media","shoe","kitchen","car","bike","toy","phone","food","sport"], verbose=True):
    print "Kaggle distance..."
    #dic = index_corpus()
    #with open("dic2.pkl", "w") as f: pickle.dump(dic, f) #dic2 encoded without median relevance
    #with open("dic3.pkl", "w") as f: pickle.dump(dic, f) #only train dic2 encoded without median relevance
    with open("dic3.pkl", "r") as f: dic = pickle.load(f)
    #print "nkd:",nkd('apple','iphone',d)
    #print "nkd:",nkd('apple','peach',d)    
    # = ["notebook","computer","movie","clothes","media","shoe","kitchen","car","bike","toy","phone","food","sport"]
    stemmer = PorterStemmer()
    
    if general_topics is None:
      n = 1
    else:
      n = len(general_topics)+1
    
    Xall_new = np.zeros((Xall.shape[0],n))
    for i in range(Xall.shape[0]):
	query = Xall["query"].iloc[i].lower()
	title = Xall["product_title"].iloc[i].lower()
	title=re.sub("[^a-zA-Z0-9]"," ", title)
	nquery = len(query.split())
		
	topics = title.split()

	#print "query:",query
	#print "title:",title
	dist_total = 0.0
	for qword in query.split():	      
	      #print "qword:",qword
	      if not qword in topics:
		bestvalue=2.0
		for tword in topics:
		    #print "qword:",qword
		    #print "tword:",tword
		    dist = nkd(qword,tword,dic)
		    #print "nkd:",dist
		    if dist<bestvalue:
		      bestvalue=dist
		dist_total += bestvalue
	      #print "nkd-best:",dist_total
	      #print "nkd_total",dist_total
	      if general_topics is not None:
		for j,topic in enumerate(general_topics):
		  dist = nkd(qword,topic,dic)
		  Xall_new[i,1+j] = Xall_new[i,1+j] + dist/float(nquery)
		  #print "qword:%s topic:%s nkd:%4.2f nkd-avg: %4.2f"%(qword,topic,dist,Xall_new[i,1+j])
		#raw_input()
	      
	Xall_new[i,0] = dist_total / float(nquery)	
       
    if general_topics is None:
      Xall_new = pd.DataFrame(Xall_new,columns=['avg_nkd'])
    else:
      Xall_new = pd.DataFrame(Xall_new,columns=['avg_nkd']+general_topics)
    print Xall_new.describe()
    #print topic_modeling(dic,topics)
    #raw_input()
    
    print "finished"
    return Xall_new
    
  
def useBenchmarkMethod(X,returnList=True,verbose=False):
    print "Create benchmark features..."
    X = X.fillna("")
    stemmer = PorterStemmer()
    s_data=[]
    for i in range(X.shape[0]):	
        s=(" ").join(["q"+ z for z in BeautifulSoup(X["query"].iloc[i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(X["product_title"].iloc[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(X["product_description"].iloc[i]).get_text(" ")      
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split()])
        s_data.append(s.lower())
	if i%5000==0:
	      print "i:",i
    if returnList:
      X = s_data
      X = pd.DataFrame(X,columns=['query']) 
    else:
      X = np.asarray(s_data)
      X = X.reshape((X.shape[0],-1))
      X = pd.DataFrame(X,columns=['concate_all']) 
    
    print "Finished.."
    #print X
    #print type(X[0])
    
    return X
    
# Use Pandas to read in the training and test data
#train = pd.read_csv("../input/train.csv").fillna("")
#test  = pd.read_csv("../input/test.csv").fillna("")

def build_query_correction_map(print_different=True):
	train = pd.read_csv('./data/train.csv').fillna("")
	test  = pd.read_csv("./data/test.csv").fillna("")
	# get all query
	queries = set(train['query'].values)
	correct_map = {}
	if print_different:
	    print("%30s \t %30s"%('original query','corrected query'))
	for q in queries:
		corrected_q = autocorrect_query(q,train=train,test=test,warning_on=False)
		if print_different and q != corrected_q:
		    print ("%30s \t %30s"%(q,corrected_q))
		correct_map[q] = corrected_q
	return correct_map

def autocorrect_query(query,train=None,test=None,cutoff=0.8,warning_on=True):
	"""
	autocorrect a query based on the training set
	"""	
	if train is None:
		train = pd.read_csv('./data/train.csv').fillna('')
	if test is None:
		test = pd.read_csv('./data/test.csv').fillna('')
	train_data = train.values[train['query'].values==query,:]
	test_data = test.values[test['query'].values==query,:]
	s = ""
	for r in train_data:
		#print "----->r2:",r[2]
		#print "r3:",r[3]
		
		s = "%s %s %s"%(s,BeautifulSoup(r[2]).get_text(" ",strip=True),BeautifulSoup(r[3]).get_text(" ",strip=True))
		#print "s:",s
		#raw_input()
	for r in test_data:
		s = "%s %s %s"%(s,BeautifulSoup(r[2]).get_text(" ",strip=True),BeautifulSoup(r[3]).get_text(" ",strip=True))
	s = re.findall(r'[\'\"\w]+',s.lower())
	#print s
	s_bigram = [' '.join(i) for i in bigrams(s)]
	#print s_bigram
	#raw_input()
	s.extend(s_bigram)
	corrected_query = []	
	for q in query.lower().split():
		#print "q:",q
		if len(q)<=2:
			corrected_query.append(q)
			continue
		corrected_word = difflib.get_close_matches(q, s,n=1,cutoff=cutoff)
		#print "correction:",corrected_word
		if len(corrected_word) >0:
			corrected_query.append(corrected_word[0])
		else :
			if warning_on:
				print ("WARNING: cannot find matched word for '%s' -> used the original word"%(q))
			corrected_query.append(q)
		#print "corrected_query:",corrected_query
		#raw_input()
	return ' '.join(corrected_query)


def autocorrect():
    query_map = build_query_correction_map()
    with open("query_map.pkl", "w") as f: pickle.dump(query_map, f)    
    