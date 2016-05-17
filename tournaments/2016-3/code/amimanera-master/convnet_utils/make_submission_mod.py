#!/usr/bin/python 
# coding: utf-8
# submission script for kaggle data science bowl

import os
import sys
import subprocess
import random
import csv
import subprocess
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#endings = ['_orig','_mod90','_mod180','_mod270','_modfliptb','_modfliplr']
endings = ['_orig','_mod90','_mod180','_mod270','_modtrans1','_modtrans2','_modresize1','_modresize2','_modshear1','_modshear2']
train_folder = '/home/loschen/programs/cxxnet/example/kaggle_bowl/data/test_last/'
predict_model='pred_model6.conf'

#endings = ['_orig','_mod180','_modfliptb','_modfliplr']
print "Suffices:",endings

def createTestList():  
    for suffix in endings:
      random.seed(888)

      task = 'test'
      
      sample = './sampleSubmission.csv'
      outfile = 'test'+suffix+'.lst'
      
      fc = csv.reader(file(sample))
      fi = train_folder
      fo = csv.writer(open(outfile, "w"), delimiter='\t', lineterminator='\n')

      # make class map
      head = fc.next()
      head = head[1:]

      # make image list
      img_lst=[]
      cnt = 0
      lst = os.listdir(fi)
      for img in lst:
	  if (suffix in endings) and (suffix in img):
	    #print img
	    img_lst.append((cnt, 0, fi + img))
	    cnt += 1
	  #if ('orig' in suffix) and (suffix not in endings[1:]): 
	  if ('orig' in suffix) and (img.find('_mod')<1):
	    img_lst.append((cnt, 0, fi + img))
	    cnt += 1

      # shuffle
      random.shuffle(img_lst)
      #wirte
      for item in img_lst:
	  fo.writerow(item)
	
def createBinaries():
      for suffix in endings:
	  bin_file = 'test'+suffix+'.bin'
	  lst_file = 'test'+suffix+'.lst'
	  call_str = "/home/loschen/programs/cxxnet/tools/im2bin "+lst_file+" \"\" "+bin_file
	  call_str = "/home/loschen/programs/cxxnet/tools/im2bin"
	  options = " "+lst_file+" \"\" "+bin_file
	  print call_str
	  subprocess.call(call_str+options,shell=True)


def make_predictions():
      for suffix in endings:
	  #rename files
	  orig_bin_file = 'test'+suffix+'.bin'
	  tmp_bin_file = 'test.bin'
	  shutil.copyfile(orig_bin_file, tmp_bin_file)
	  orig_lst_file = 'test'+suffix+'.lst'
	  tmp_lst_file = 'test.lst'
	  shutil.copyfile(orig_lst_file, tmp_lst_file)
	  call_str = "../../bin/cxxnet "+predict_model
	  subprocess.call(call_str,shell=True)
	  #rename test.txt
	  out_file = 'test'+suffix+'.txt'
	  os.rename('test.txt',out_file)
	  
def format_predictions():
    for suffix in endings:
	sample = './sampleSubmission.csv'
	orig_lst_file = 'test'+suffix+'.lst'
	pred_file = 'test'+suffix+'.txt'
	sub_file = 'cxx_'+suffix+'.csv'
	
	fc = csv.reader(file(sample))
	fl = csv.reader(file(orig_lst_file), delimiter='\t', lineterminator='\n')
	fi = csv.reader(file(pred_file), delimiter=' ', lineterminator='\n')
	fo = csv.writer(open(sub_file, "w"), lineterminator='\n')
	
	head = fc.next()
	fo.writerow(head)

	#head = head[1:]

	img_lst = []
	for line in fl:
	    path = line[-1]
	    path = path.split('/')
	    path = path[-1]
	    path = path.replace(suffix,'')
	    img_lst.append(path)

	idx = 0
	for line in fi:
	    row = [img_lst[idx]]
	    idx += 1
	    line = line[:-1]
	    row.extend(line)
	    fo.writerow(row)

def average_predictions(basename='cxx_',lendings=endings,weights=None):
    print weights
    if weights is not None:
	if len(weights) <> len(lendings):
	  print "Weights and ending do not match."
	  sys.exit(1)
	weights = weights/np.sum(weights)
	print "Weights:", weights
	
      
    sample = basename+lendings[0]+'.csv'
    ref = pd.read_csv(sample, sep=",", na_values=['?'],index_col=0)
    nrows = ref.shape[0]
    #2*nsamples*nclasses
    preds_all = np.zeros((len(lendings),nrows,ref.shape[1]))
    for i,suffix in enumerate(lendings):
	sub_file = basename+suffix+'.csv'
	tmp = pd.read_csv(sub_file, sep=",", na_values=['?'],index_col=0)
	tmp = tmp.clip(1E-15,1.0-1E-15)
	tmp.sort_index(inplace=True)
	#print tmp.describe()
	if weights is not None:
	  print "Suffix:",suffix," weight:",weights[i]
	  preds_all[i] = weights[i]*tmp.values
	else:
	  preds_all[i] = tmp.values
	
	#ax1 = tmp.iloc[:,2:10].hist(bins=40)
	#plt.title('tmp_'+str(i))
	#raw_input()
    
    if weights is not None:
      preds = np.sum(preds_all,axis=0)
    else:
      preds = np.mean(preds_all,axis=0)
    print "Final shape:",preds.shape
    
    preds = pd.DataFrame(preds,index=tmp.index,columns=tmp.columns)
    #clip it
    preds = preds.where(preds>1E-15,1E-15).where(preds<(1.0-1E-15),(1.0-1E-15))

    #print preds.describe()
  
    filename = basename+'blend4.csv'
    filename = '/home/loschen/calc/amimanera/competition_data/submissions/'+filename
    preds.to_csv(filename,index_label='image')
    
    ax1 = ref.iloc[:,2:10].hist(bins=40)
    plt.title('ref')
    ax2 = preds.iloc[:,2:10].hist(bins=40)
    plt.show()


def test_submissions(clip=True):
    #for i,suffix in enumerate(endings):
#	sub_file = 'cxx_'+suffix+'.csv'
#	tmp = pd.read_csv(sub_file, sep=",", na_values=['?'],index_col=0)
#	print tmp.describe().iloc[:,1:5]
#	raw_input()
	
    filename = 'sampleSubmission.csv'
    filename = '/home/loschen/calc/amimanera/competition_data/submissions/'+filename
    tmp = pd.read_csv(filename, sep=",", na_values=['?'],index_col=0)
    print tmp.describe().iloc[:,1:5]
    if clip:
	tmp = tmp.clip(1E-15,1.0-1E-15)
	tmp.to_csv(filename,index_label='image')	
	
if __name__=="__main__":
    #createTestList()
    #createBinaries()
    #make_predictions()
    #format_predictions()
    #average_predictions(lendings=endings)
    #average_predictions(lendings=endings,weights=[0.2,0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05])
    average_predictions(lendings=['model5_avg','model6_avg'],weights=[0.25,0.75])
    #average_predictions(lendings=['large_rot_mean','addlayer2_avg'],weights=None)
    #average_predictions()
    test_submissions(clip=False)
