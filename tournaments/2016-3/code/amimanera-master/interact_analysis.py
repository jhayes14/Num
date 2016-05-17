# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:31:15 2013

@author: loschen
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

import readline


def interact_analysis(df,n_bins=100):

    #plt.ion()
    
    readline.parse_and_bind('tab: complete')
    readline.parse_and_bind('set editing-mode vi')
    
    usage = """
            (n) next
            (p) plot
            (l) logtransform
            (s) sqrt transform
            (r) reset series
            (q) quit
            """
    func_dic = {'np.log1p': np.log1p, 'np.sqrt': np.sqrt}
    func_hist={}
    print df.columns
    num_columns = df._get_numeric_data().columns
    print "\n\nInteractive Analysis of Data\n"
    print "Numeric columns       n=%4d %s"%(len(num_columns),num_columns)
    nonnum_columns = df.columns.difference(num_columns)
    print "Non-numeric columns   n=%4d %s"%(len(nonnum_columns),nonnum_columns)

    for i,col in enumerate(num_columns):
        leave_it = False
        df_orig = df[col].copy()
        func_hist[col] = None
        while True:
	    print "Column:",col
            print df[col].describe()
            line = raw_input(usage)
            print 'ENTERED: "%s"' % line
            if line == 'q':
                leave_it = True
                func_hist[col] = None
                break
            if line == 'l':
                df[col] = df[col].map(func_dic['np.log1p'])
                func_hist[col] = 'np.log1p'
                line = 'p'
            if line == 's':
                df[col] = df[col].map(func_dic['np.sqrt'])
                func_hist[col] = 'np.sqrt'
                line = 'p'
            if line == 'n':
                break
            if line == 'r':
                df[col] = df_orig
                func_hist[col] = None
                line = 'p'
            if line in num_columns:
		col = line
		df_orig = df[col].copy()
            print "Column:",col
            print df[col].describe()
            
            if line == 'p':
                plt.close()
                plt.figure(1)
                plt.subplot(211)
                plt.hist(df_orig.values,bins=n_bins)
                
                plt.subplot(212)
                plt.hist(df[col].values,bins=n_bins)
                plt.xlabel(col)
                plt.show()
        
        if leave_it == True: break
    
    for key in func_hist.keys():
	print "Col: %s  transformation: %s"%(key,func_hist[key])
    
    print "Finished!"
    

def pairplot(df,hue_name=None):

    if hue_name is not None:
        df['target']=hue_name
        sns.pairplot(df, hue='target')
    else:
        sns.pairplot(df)


if __name__=="__main__":
    np.random.seed(123)
    plt.style.use('ggplot')
    df = pd.read_csv("D:\COSMOlogic\logK\logk_qspr\logK(octanol_water)_corr.csv",sep=";")
    interact_analysis(df)