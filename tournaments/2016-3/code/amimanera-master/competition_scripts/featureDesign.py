#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  crawl data
"""

import pandas as pd
import nltk as nltk
from nltk.tag import pos_tag
#from nltk.tag.simplify import simplify_wsj_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.probability import LidstoneProbDist
import random
#from nltk.stem.wordnet import WordNetStemmer
from nltk.stem.porter import PorterStemmer

def getStopwords(addRecipe=True):
    stop_words=None
    #adapted from http://blog.kaggle.com/2012/07/17/getting-started-with-the-wordpress-competition/
    #stop_words = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","to","too","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your"]   
    #recipe_words=["recipe","food","meal","kitchen","cook","apetite","meal","cuisine","cake","baking","apple","sweet","cookie","brownie","chocolat"]
    #recipe_words=["recipe"]
    if addRecipe:
	for x in recipe_words:
	    stop_words.append(x)
    print stop_words
    return(stop_words)


def posTagging(olddf):
    """
    Creates new features
    """
    print "Use nltk postagging..."
    tutto=[]
    taglist=['N','NP','ADJ','ADV','PRO','V','NUM','VD','DET','P','WH','MOD','TO','VG','CNJ']
    #taglist=['N','NP','ADJ','ADV']
    
    #olddf = olddf.ix[random.sample(olddf.index, 10)]
    olddf=pd.DataFrame(olddf['body'])
    
    print type(olddf)
    for ind in olddf.index:
	  print ind
	  row=[]
	  row.append(ind)
	  text=olddf.ix[ind,'body']
	  tagged=pos_tag(word_tokenize(text))
	  tagged = [(word, simplify_wsj_tag(tag)) for word, tag in tagged]
	  
	  tag_fd = FreqDist(tag for (word, tag) in tagged)
	  
	  print tagged
	  #print len(tagged)
	  
	  for l in taglist:
	      f= tag_fd[l]/float(len(tagged))
	      #print f
	      row.append(f)
	
	  #tag_fd.plot(cumulative=False)
	  #raw_input("HITKEY")
    
    
    #for index,row in pd.DataFrame(olddf['body']).iterrows():
	#tagged=pos_tag(word_tokenize(str(row)))
	#tag_fd = FreqDist(tag for (word, tag) in tagged)
	#print tag_fd.keys()
	
	#tag_fd.plot(cumulative=True)
	  tutto.append(row)
    newdf=pd.DataFrame(tutto).set_index(0)
    newdf.columns=taglist
    print newdf.head(20)
    print newdf.describe()
    newdf.to_csv("../stumbled_upon/data/postagged2.csv")

    
def postagFeatures(olddf):
    """
    Creates new features
    """
    print "Create POS tag as features..."
    tutto=[]
    olddf=pd.DataFrame(olddf['body'])
    for ind in olddf.index:
	  print "postag feats: ",ind
	  row=[]
	  row.append(ind)
	  text=olddf.ix[ind,'body']
	  tagged=pos_tag(word_tokenize(text))
	  tagged = [(word, simplify_wsj_tag(tag)) for word, tag in tagged]
	  posbody=""
	  for word, tag in tagged:
	    posbody=posbody+tag+" "
	  posbody=posbody+"."
	  row.append(posbody)
	  #print row
	
	  #tag_fd.plot(cumulative=False)
	  #raw_input("HITKEY")
    
    
    #for index,row in pd.DataFrame(olddf['body']).iterrows():
	#tagged=pos_tag(word_tokenize(str(row)))
	#tag_fd = FreqDist(tag for (word, tag) in tagged)
	#print tag_fd.keys()
	#tag_fd.plot(cumulative=True)
	  tutto.append(row)
    newdf=pd.DataFrame(tutto).set_index(0)
    newdf.columns=["postagfeats"]
    print newdf.head(20)
    print newdf.describe()
    newdf.to_csv("../stumbled_upon/data/postagfeats_simple.csv",encoding="utf-8")
    
def postagSmoothing(olddf):
    """
    Use postag to smooth data
    """
    print "Smoothing data via postag..."
    tutto=[]    
    #olddf = olddf.ix[random.sample(olddf.index, 10)]
    #olddf = olddf.ix[olddf.index[0:9]]
    olddf=pd.DataFrame(olddf['body'])
    wnl = PorterStemmer()
    for ind in olddf.index:
	  print "Smoothing: ",ind
	  row=[]
	  row.append(ind)
	  text=olddf.ix[ind,'body']
	  text=[wnl.stem(t) for t in word_tokenize(text)]	  
	  tagged=pos_tag(text)
	  word_tag=u''
	  tag_word=u''
	  actual_tag=u'.'
	  actual_word=u'.'
	  #print tagged
	  for word, tag in tagged:
	      #tag=simplify_wsj_tag(tag)
	      word=word.lower()
	      word_tag=word_tag+actual_word+u"_"+tag+" "
	      actual_word=word
	      tag_word=tag_word+actual_tag+u"_"+word+" "
	      actual_tag=tag
	  #print word_tag
	  #print tag_word
	  row.append(word_tag)
	  row.append(tag_word)
	  tutto.append(row)
    newdf=pd.DataFrame(tutto).set_index(0)
    newdf.columns=[u'wordtag',u'tagword']
    print newdf.head(20)
    print newdf.describe()
    newdf.to_csv("../stumbled_upon/data/postagsmoothed2.csv",encoding="utf-8")
    
    
def lidstoneProbDist(olddf):
    """
    Use nltk to create probdist
    """
    #http://www.inf.ed.ac.uk/teaching/courses/icl/nltk/probability.pdf
    #https://github.com/tuzzeg/detect_insults/blob/master/README.md
    print "Creating LidStone Probdist...",nltk.__version__
    tutto=[]
    
    #olddf = olddf.ix[random.sample(olddf.index, 10)]
    olddf=pd.DataFrame(olddf['body'])
    
    print type(olddf)
    for ind in olddf.index:
	  print ind
	  row=[]
	  row.append(ind)
	  text=olddf.ix[ind,'body']
	  tokens=word_tokenize(text)
	  #print tokens
	  
	  t_fd = FreqDist(tokens)
	  pdist = LidstoneProbDist(t_fd,0.1)
	  print pdist.samples()
	  #for tok in tokens:
	  #    print pdist[3][tok]
	  #t_fd.plot(cumulative=False)
	  raw_input("HITKEY")
	  row=tokens
	  #print tagged
	  #print len(tagged)

	  tutto.append(row)
    newdf=pd.DataFrame(tutto).set_index(0)
    newdf.columns=taglist
    print newdf.head(20)
    print newdf.describe()
    newdf.to_csv("../stumbled_upon/data/lidstone.csv")
    
    
def featureEngineering(olddf):
    """
    Creates new features
    """
    print "Feature engineering..."
    #lower
    olddf['url']=olddf.url.str.lower()
    olddf['embed_ratio']=olddf.embed_ratio.replace(-1.0,0.0)
    olddf['image_ratio']=olddf.image_ratio.replace(-1.0,0.0)
    olddf['is_news']=olddf['is_news'].fillna(0)
    olddf['news_front_page']=olddf['news_front_page'].fillna(0)
    #url length
    tmpdf=olddf.url.str.len()
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['url_length']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
    #body plate length
    tmpdf=olddf.body.str.len()
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['body_length']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
    
    tmpdf=olddf.title.str.len()
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['title_length']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
    
    #tmpdf=olddf.title.str.split().len()
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['title_word_count']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    
    #boiler plate length
    #tmpdf=olddf.boilerplate.str.len()
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['boilerplate_length']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #counts exclamation marks
    #tmpdf=olddf.body.str.count('!')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['excl_mark_number']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #counts exclamation marks
    #tmpdf=olddf.body.str.count('\\?')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['quest_mark_number']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains .com
    #tmpdf=olddf.url.str.contains('\.com')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_com']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains .org
    #tmpdf=olddf.url.str.contains('\.org')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_org']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains .co.uk
    #tmpdf=olddf.url.str.contains('\.co\.uk')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_co_uk']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
     #endswith .com
    #tmpdf=olddf.url.str.contains('com.$')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_endswith_com']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains blog
    #tmpdf=olddf.url.str.contains('blog')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_blog']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
     #contains recipe & co.
    tmpdf=olddf.url.str.contains('recipe|food|meal|kitchen|cook|apetite|meal|cuisine')
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['url_contains_foodstuff']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
     #contains recipe & co.
    tmpdf=olddf.url.str.contains('recipe')
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['url_contains_recipe']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains sweet stuff
    tmpdf=olddf.url.str.contains('cake|baking|apple|sweet|cookie|brownie|chocolat')
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['url_contains_sweetstuff']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains diet
    #tmpdf=olddf.url.str.contains('diet|calorie|nutrition|weight|fitness')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_dietfitness']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
     #contains recipe & co.
    #tmpdf=olddf.url.str.contains('recipe')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_recipe']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
     #contains health
    tmpdf=olddf.url.str.contains('health|fitness|exercise')
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['url_contains_health']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains www
    #tmpdf=olddf.url.str.contains('www')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_www']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains news
    tmpdf=olddf.url.str.contains('news|cnn')
    tmpdf=pd.DataFrame(tmpdf.astype(int))
    tmpdf.columns=['url_contains_news']
    #print tmpdf.describe()
    olddf= pd.concat([olddf, tmpdf],axis=1)
    #contains obscene
    #tmpdf=olddf.url.str.contains('obscene|sex|nude|fuck|asshole')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_obscene']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #girls
    #tmpdf=olddf.url.str.contains('girls|nude|sex|nipple')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['url_contains_girls']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    #syria
    #tmpdf=olddf.boilerplate.str.contains('syria|damaskus|bashar|assad')
    #tmpdf=pd.DataFrame(tmpdf.astype(int))
    #tmpdf.columns=['body_contains_syria']
    #print tmpdf.describe()
    #olddf= pd.concat([olddf, tmpdf],axis=1)
    return olddf