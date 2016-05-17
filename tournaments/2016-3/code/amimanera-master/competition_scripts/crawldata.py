#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""  crawl data
"""

import pandas as pd
import re
import math

def crawlHTML(lXall):
      """
      crawling raw data
      """
      print "Crawling html data..."
      basedir='../stumbled_upon/raw_content/'
      #phtml = re.compile("</[^>]*?>")
      phtml = re.compile("<[^>]*?>")
      tutto=[]
      for ind in lXall.index:
	  row=[]
	  #nl=lXall.ix[ind,'numberOfLinks']
	  #nl=1+lXall.ix[ind,'non_markup_alphanum_characters']
	  #print "numberOfLinks:",nl
	  with open(basedir+str(ind), 'r') as content_file:
	    content = content_file.read()
	    #print "id:",ind,
	    row.append(ind)
	    
	    res = phtml.findall(content)
	    tmp=[x for x in res]
	    tmp=tmp[:100]
	    tmp=' '.join(tmp)
	    tmp=unicode(tmp, errors='replace')
	    tmp=tmp.lower()
	    tmp=tmp.replace("<","").replace(">","").replace("/","") 
	    #tmp=tmp.decode("utf8")
	    #print tmp
	    row.append(tmp)
	    
	    #if len(res)>0:
		#print ind,": ",res
		#raw_input("HITKEY")
		
	  tutto.append(row)
      newdf=pd.DataFrame(tutto).set_index(0)
      newdf.columns=['htmltag']
      print newdf.head(20)
      print newdf.describe()
      return newdf


def crawlRawData(lXall):
      """
      crawling raw data
      """
      print "Crawling raw data..."
      basedir='../stumbled_upon/raw_content/'
      pfacebook = re.compile("www.{1,2}facebook.{1,2}com")
      pfacebook2 = re.compile("developers.{1,2}facebook.{1,2}com.{1,2}docs.{1,2}reference.{1,2}plugins.{1,2}like|facebook.{1,2}com.{1,2}plugins.{1,2}like")
      plinkedin = re.compile("platform.{1,2}linkedin.{1,2}com")
      ptwitter = re.compile("twitter.{1,2}com.{1,2}share")
      prss=re.compile("rss feed",re.IGNORECASE)
      pgooglep=re.compile("apis.{1,2}google.{1,2}com")
      #pstumble=re.compile("www.{1,2}stumbleupon.{1,2}com")
      pstumble=re.compile("stumbleupon")
      pcolor=re.compile("colorscheme|color_scheme|color=|color:",re.IGNORECASE)
      psignup=re.compile("signup|register|login|sign up",re.IGNORECASE)
      pcomment=re.compile("leave a comment|leave comment",re.IGNORECASE)
      pncomment=re.compile("comment-",re.IGNORECASE)
      pmail=re.compile("email",re.IGNORECASE)
      ppics=re.compile("\.png|\.tif|\.jpg",re.IGNORECASE)
      pgif=re.compile("\.gif",re.IGNORECASE)
      psmile=re.compile(":-\)|;-\)")
      plbreak=re.compile("<br>")
      psearch=re.compile("searchstring|customsearch|searchcontrol|searchquery|searchform|searchbox",re.IGNORECASE)
      pcaptcha=re.compile("captcha",re.IGNORECASE)
      padvert=re.compile("advertis",re.IGNORECASE)
      pnewline=re.compile("\n")
      pgooglead=re.compile("google_ad_client")
      phtml5=re.compile("html5",re.IGNORECASE)
      phuff=re.compile("www.huffingtonpost.com",re.IGNORECASE)
      pflash=re.compile("shockwave-flash",re.IGNORECASE)
      pdynlink=re.compile("<a href.+?.+>")
      pnofollow=re.compile("rel=\"nofollow\"",re.IGNORECASE)
      pschemaorg=re.compile("schema\.org",re.IGNORECASE)
      pmobileredirect=re.compile("mobile redirect",re.IGNORECASE)
      
      #pshare=re.compile("sharearticle|share.{1,20}article",re.IGNORECASE)
      plang=re.compile("en-US|en_US",re.IGNORECASE)
      tutto=[]
      for ind in lXall.index:
	  row=[]
	  nl=1.0+lXall.ix[ind,'numberOfLinks']
	  nchar=1.0+lXall.ix[ind,'non_markup_alphanum_characters']
	  #print "numberOfLinks:",nl
	  with open(basedir+str(ind), 'r') as content_file:
	    content = content_file.read()
	    #print "id:",ind,
	    row.append(ind)
	    
	    res = pfacebook.findall(content)
	    row.append(len(res)/float(nl))
	    
	    res = pfacebook2.findall(content)	    
	    row.append(len(res)/float(nl))
	    
	    res = ptwitter.findall(content)
	    row.append(len(res)/float(nl))
	
	    
	    #res = prss.findall(content)
	    #row.append(len(res)/float(nl))
	    
	    #res = pgooglep.findall(content)	    
	    #row.append(len(res)/float(nl))
	    
	    #res = pstumble.findall(content)	    
	    #row.append(len(res)/float(nl))
	    
	    res = pncomment.findall(content)	    
	    row.append(len(res))
	    
	    #res = pcolor.findall(content)	    
	    #row.append(len(res))
	    
	    #res = psmile.findall(content)	    
	    #row.append(len(res))
	    
	    #if len(res)>0:
		#print ind,": ",res
		#raw_input("HITKEY")
	    
	    #res = plbreak.findall(content)	    
	    #row.append(len(res))
	    
	    #res = padvert.findall(content)	    
	    #row.append(len(res))
	    
	    res = pnewline.findall(content)	    
	    row.append(math.log(1.0+len(res)))
	    
	    #res = pdynlink.findall(content)	    
	    #row.append(len(res))
	    
	    #res = pnofollow.findall(content)	    
	    #row.append(len(res))
	    
	    #res = pschemaorg.findall(content)	    
	    #row.append(len(res))
	    
	    #res = pmobileredirect.findall(content)	    
	    #row.append(len(res))
	    
	    
	    
	    #m = pgooglead.search(content)
	    #if m:
	#	row.append(1)
	 #   else:
	#	row.append(0)
	    
	    #if len(res)>0:
		#print ind,": ",res
		#raw_input("HITKEY")

		
	    #res = pshare.findall(content)
	    #row.append(len(res)/float(nl))
	  #print ""
	  tutto.append(row)
      newdf=pd.DataFrame(tutto).set_index(0)
      newdf.columns=['wwwfacebook_ratio','facebooklike_ratio','twitter_ratio','n_comment','logn_newline']
      pd.set_printoptions(max_rows=40, max_columns=20)
      print newdf.head(20)
      print newdf.describe()
      return newdf