#!/usr/bin/python 
# coding: utf-8
import numpy as np
import pandas as pd
import sklearn as sl
import random
import math

from qsprLib import *
import inspect
import pickle

from pandas.tools.plotting import scatter_matrix
from xgboost_sklearn import *


def createFeatures(X_all,keepAll=True,createNAFeats='all'):
    """
    Create some new features
    
    keep createNAFeats
    
    """
    def f(x):
        if x==-999.0:
            return 1.0
        else:
            return 0.0    

    print "NA Feature creation..."
    X_NA=pd.DataFrame(index=X_all.index)

    for colname in X_all.columns:
        if (X_all[colname]==-999.0).any():
            new_name=colname+'_NA'
            if 'all' not in createNAFeats:
		if new_name in createNAFeats:
		    X_NA[new_name]=X_all[colname].map(f)
	    else:
		X_NA[new_name]=X_all[colname].map(f)
		
            #X_all['hasNA']=X_all[colname].map(f2)
    if keepAll:
        X_all = pd.concat([X_all, X_NA],axis=1)
    else:
        X_NA['NA_sum'] = X_NA.sum(axis=1)  
	X_all = pd.concat([X_all, X_NA['NA_sum']],axis=1)
	
    print X_all.columns
    print "End of feature creation..."    
    
    return (X_all)


def massImputer(X_orig,y_tmp,massmodel,doSVD=None,nsamples=250000,newfeature=True,loadData=True):
    """
    try to learn missing DER_mass_MMC from other data
    """
    print "Imputing mass..."
    print X_orig['DER_mass_MMC'].describe()
    if loadData or 'load' in massmodel:
	X_orig['DER_mass_EST'] = pd.read_csv('../datamining-kaggle/higgs/mass_est.csv', sep=",", na_values=['?'], header=None, index_col=0)
	print X_orig['DER_mass_EST'].describe()
	return(X_orig)
   
    y_tmp=pd.DataFrame({'y' : pd.Series(y_tmp)})
    
    print massmodel
    
    idx_NA = np.asarray(X_orig['DER_mass_MMC']) < -998.0
    print type(idx_NA)
    y_mass = np.asarray(X_orig.ix[-idx_NA,'DER_mass_MMC'])
    X_tmp = X_orig.drop(['DER_mass_MMC'], axis=1)
    #here we should impute other values
    #X_tmp= X_tmp.fillna(X_tmp.mean())
    X_tmp= np.asarray(X_tmp)
    #make SVD to improve performance
    if doSVD is not None:
	tsvd=TruncatedSVD(n_components=doSVD, algorithm='randomized', n_iter=5, tol=0.0)
	X_tmp=tsvd.fit_transform(X_tmp)
    scaler=StandardScaler()
    X_tmp = scaler.fit_transform(X_tmp)
    
    #X_svd=pd.DataFrame(np.asarray(X_svd),index=X_all.index)
    X_mass = X_tmp[-idx_NA]
    X_mass_test = X_tmp[idx_NA]
      
    print "Fitting mass model with subset of length:",nsamples
    if nsamples != -1: 
	rows = random.sample(np.arange(X_mass.shape[0]), nsamples)
	X_mass = X_mass[rows]
	y_mass = y_mass[rows]
	
    print "Dim X_mass:",X_mass.shape
    #print "Dim X_test:",X_mass_test.shape
    print "Dim y:",y_mass.shape
    
    massmodel.fit(X_mass,y_mass)
    print "Prediction..."
    mass_pred = massmodel.predict(X_mass)
    print mass_pred
    print y_mass
    mse = mean_squared_error(y_mass, mass_pred)
    print "MSE=%6.3f RMSE=%6.3f"%(mse,np.sqrt(mse))
    sct = plt.scatter(mass_pred, y_mass, c=y_tmp['y'],s=50,linewidths=2, edgecolor='black')
    #plt.plot(mass_pred, y_mass,'ro')
    sct.set_alpha(0.75)
    plt.show()
    
    #mass_pred_test = massmodel.predict(X_mass_test)
    X_mass = X_orig.drop(['DER_mass_MMC'], axis=1)
    if doSVD is not None:
	X_mass=tsvd.fit_transform(X_mass)
    
    print "Dim X_mass:",X_mass.shape
    
    if newfeature:
	X_orig['DER_mass_EST']=massmodel.predict(X_mass)
	X_orig['DER_mass_EST'].to_csv("../datamining-kaggle/higgs/mass_est.csv")
    else:
	X_orig.ix[idx_NA,'DER_mass_MMC']=massmodel.predict(X_mass_test)
    
    print "Dim y:",y_mass.shape
    print "Dim X:",X_mass.shape
    print "Dim X_test:",X_mass_test.shape
    print "Dim X_orig:",X_orig.shape
    
    print X_orig['DER_mass_EST'].describe()
    print X_orig['DER_mass_MMC'].describe()
    
    return(X_orig)
    
    
def massEstimator(X_all,createPsq=True,normalize=True,invertEta=True):
    """
    Create features according to:
    M_tt =m_vis / x1 x2
    with x1,2 =p_vis1,2 /(p_vis1,2 + p_mis1,2 )

    Eq. 2 in Elagin et al.
    """
    print "Mass estimator & feature creation..."
    
    if createPsq:
	tmp = X_all['PRI_tau_phi'].map(np.cos)
	X_all['p_tau_x']=X_all.PRI_tau_pt*tmp
	tmp = X_all['PRI_tau_phi'].map(np.sin)
	X_all['p_tau_y']=X_all.PRI_tau_pt*tmp
	tmp = X_all['PRI_tau_eta'].map(np.sinh)
	X_all['p_tau_z']=X_all.PRI_tau_pt*tmp
	
	#X_all['p_tau_abs']=X_all['p_tau_x']*X_all['p_tau_x']+X_all['p_tau_y']*X_all['p_tau_y']+X_all['p_tau_z']*X_all['p_tau_z']
	#X_all['p_tau_abs']=X_all['p_tau_abs'].map(np.sqrt)
	
    
      
	tmp = X_all['PRI_lep_phi'].map(np.cos)
	X_all['p_lep_x']=X_all.PRI_lep_pt*tmp
	tmp = X_all['PRI_lep_phi'].map(np.sin)
	X_all['p_lep_y']=X_all.PRI_lep_pt*tmp
	tmp = X_all['PRI_lep_eta'].map(np.sinh)
	X_all['p_lep_z']=X_all.PRI_lep_pt*tmp
	
	#X_all['p_lep_abs']=X_all['p_lep_x']*X_all['p_lep_x']+X_all['p_lep_y']*X_all['p_lep_y']+X_all['p_lep_z']*X_all['p_lep_z']
	#X_all['p_lep_abs']=X_all['p_lep_abs'].map(np.sqrt)
	
	#new features accoring to eq. 1 sasonov et al.
	#X_all['p_tauXp_lep']=X_all['p_tau_abs']*X_all['p_lep_abs']
	#X_all['metXp_tau']=X_all['PRI_met']*X_all['p_tau_abs']
	#X_all['metXp_lep']=X_all['PRI_met']*X_all['p_lep_abs']#!
	
	X_all['p_lepXp_tau_vec']=X_all['p_lep_x']*X_all['p_tau_x']+X_all['p_lep_y']*X_all['p_tau_y']+X_all['p_lep_z']*X_all['p_tau_z']#!
	X_all['metXp_tau_vec']=X_all['PRI_met']*X_all['PRI_met_phi'].map(np.cos)*X_all['p_tau_x']+X_all['PRI_met']*X_all['PRI_met_phi'].map(np.sin)*X_all['p_tau_y']#!
	X_all['metXp_lep_vec']=X_all['PRI_met']*X_all['PRI_met_phi'].map(np.cos)*X_all['p_lep_x']+X_all['PRI_met']*X_all['PRI_met_phi'].map(np.sin)*X_all['p_lep_y']#!
    
	del X_all['p_lep_x']
	del X_all['p_lep_y']
	del X_all['p_lep_z']
	del X_all['p_tau_x']
	del X_all['p_tau_y']
	del X_all['p_tau_z']
    
    
    if normalize:
	delta_angle = 'PRI_tau_phi'
	for angle in ['PRI_lep_phi', 'PRI_met_phi'] :
	    label='%s-%s' % (angle, delta_angle)
	    X_all[label] = delta_angle_norm(X_all[angle], X_all[delta_angle])
	    #plt.hist(X_all[label].values,label=label,bins=40)
	    #plt.legend()
	    #plt.show()
	    #del X_all[angle]
	
	for angle in ['PRI_jet_leading_phi', 'PRI_jet_subleading_phi'] :
	    X_all['%s-%s'% (angle, delta_angle)] = (
		delta_angle_norm(X_all[angle], X_all[delta_angle]) * (X_all[angle] != -999) +
		(-999) * (X_all[angle] == -999))
	    del X_all[angle]
	
	del X_all[delta_angle]

    if invertEta:
	invert_mask = X_all['PRI_tau_eta'] < 0
	for angle in ['PRI_tau_eta', 'PRI_lep_eta', 'PRI_jet_leading_eta', 'PRI_jet_subleading_eta'] :
	    X_all[angle] = angle_invert(X_all[angle], invert_mask)
    
    #plt.show()
    #plt.hist(X_all['PRI_tau_phi'].values,bins=40)
    #plt.show()
    
    
    #plt.hist(tmp.values,bins=50)
    #plt.show()
    
    
def prepareDatasets(nsamples=-1,onlyPRI=False,replaceNA=True,plotting=True,stats=True,transform=False,createNAFeats=None,dropCorrelated=True,scale_data=False,clusterFeature=False,dropFeatures=None,polyFeatures=None,createMassEstimate=False,imputeMassModel=None,featureFilter=None,onehotenc=None,quadraticFeatures=None,loadCake=False):
    """
    Read in train and test data, create data matrix X and targets y   
    """
    X = pd.read_csv('../datamining-kaggle/higgs/training.csv', sep=",", na_values=['?'], index_col=0)
    X_test = pd.read_csv('../datamining-kaggle/higgs/test.csv', sep=",", na_values=['?'], index_col=0)
    
    if loadCake:
	cake = pd.read_csv('../datamining-kaggle/higgs/training-public-onlyAB.csv', sep=",", na_values=['?'], index_col=0)
	cake_test = pd.read_csv('../datamining-kaggle/higgs/test-public-onlyAB.csv', sep=",", na_values=['?'], index_col=0)
	#cake.hist(bins=100)
	#cake_test.hist(bins=100)
	#plt.show()
	#print cake_test.describe()
	X = pd.concat([X, cake],axis=1)
	X_test = pd.concat([X_test, cake_test],axis=1)

    if nsamples != -1: 
	rows = random.sample(X.index, nsamples)
	X = X.ix[rows] 
	
	#X_test = X_test.iloc[0:100,:]
    
    weights = np.asarray(X['Weight'])
    
    sSelector = np.asarray(X['Label']=='s')
    bSelector = np.asarray(X['Label']=='b')
    
    s = np.sum(weights[sSelector])  
    #b = np.sum(weights[bSelector])
    
    sumWeights = np.sum(weights)
    print "Sum weights: %8.2f"%(sumWeights)
    sumSWeights = np.sum(weights[sSelector])
    sumBWeights = np.sum(weights[bSelector])
    print "Sum (w_s): %8.2f n(s): %6d"%(sumSWeights,np.sum(sSelector==True))
    print "Sum (w_b): %8.2f n(b): %6d"%(sumBWeights,np.sum(bSelector==True))
    
    print "Unique weights for signal    :",np.unique(weights[sSelector])
    print "Unique weights for background:",np.unique(weights[bSelector])
    
    ntotal=250000
    wFactor = 1.* ntotal / X.shape[0]
    print "AMS,max: %4.3f (wfactor=%4.3f)" % (AMS(s*wFactor, 0.0),wFactor)
    
    y = X['Label'].str.replace(r's','1').str.replace(r'b','0')
    y = np.asarray(y.astype(float))   	  
        
    X = X.drop(['Weight'], axis=1)
    X = X.drop(['Label'], axis=1)
    
    #modifications for ALL DATA
    X_all = pd.concat([X_test, X])    
    
    if onehotenc is not None:
	print "One hot encoding of: ",onehotenc
	for col in onehotenc:
	    X_new = pd.get_dummies(X_all.loc[:,col],prefix=col)
	    X_all = X_all.drop([col], axis=1)
	    X_all = pd.concat([X_all, X_new],axis=1)
    
    if dropFeatures is not None:
	for col in dropFeatures:
            print "Dropping features: ",col
            X_all = X_all.drop([col], axis=1)
    
    if quadraticFeatures is not None:
        print "Creating squared features"
        X_pri = X_all.copy()
	cols = X_pri.columns
	for col in cols:
	    if col.startswith('DER') or 'jet' in col:
	      X_pri.drop([col], axis=1,inplace=True)
	      
	cols = X_pri.columns   
	n=len(X_pri.columns)
	for i,col1 in enumerate(cols):
	    new_name=col1+"_^2"
	    if new_name in quadraticFeatures:
		X_new = pd.DataFrame(X_pri.ix[:,col1].mul(X_pri.ix[:,col1]))
		X_new.columns=[new_name]
		X_all = pd.concat([X_all, X_new],axis=1)
		#delete due to correlations
		del X_all[col1]
	    
	print X_all.columns
	print "New dim:",X_all.shape
        
    
    if polyFeatures is not None:
	print "Creating feature interaction for primary features"
	
	#removing derived features
	X_pri = X_all.copy()
	cols = X_pri.columns
	for col in cols:
	    if col.startswith('DER') or 'jet' in col:
	      X_pri.drop([col], axis=1,inplace=True)
	    #if col.startswith('PRI'):
	      #X_der.drop([col], axis=1,inplace=True)
	      
	cols = X_pri.columns   
	n=len(X_pri.columns)
	for i,col1 in enumerate(cols):
	    for j,col2 in enumerate(cols):
		if j<=i: continue
		new_name=col1+"X"+col2
		if new_name in polyFeatures:
		    X_new = pd.DataFrame(X_pri.ix[:,col1].mul(X_pri.ix[:,col2]))
		    X_new.columns=[new_name]
		    X_all = pd.concat([X_all, X_new],axis=1)

	print X_all.columns
	print "New dim:",X_all.shape
	
    if imputeMassModel is not None:
	#X_all=massImputer(X_all[len(X_test.index):],y,imputeMassModel)
	X_all=massImputer(X_all,y,imputeMassModel)
    
    if createMassEstimate:
        #TODO not yet finishe
	massEstimator(X_all)
    
    if ('DER' or 'PRI') in onlyPRI:
	cols = X_all.columns
	for col in cols:
	    if col.startswith(onlyPRI):
             print "Dropping column: ",col
             X_all = X_all.drop([col], axis=1)

    if createNAFeats is not None:
        X_all=createFeatures(X_all,True,createNAFeats)
		
    if replaceNA:
	X_all = X_all.replace(-999, np.NaN)
	X_all = X_all.fillna(X_all.mean())   
       
    transcols_PRI=['PRI_met','PRI_lep_pt','PRI_met_sumet','PRI_jet_all_pt','PRI_jet_leading_pt','PRI_jet_subleading_pt','PRI_tau_pt']
    transcols_DER=['DER_mass_transverse_met_lep','DER_mass_vis','DER_pt_h','DER_pt_ratio_lep_tau','DER_pt_tot','DER_sum_pt']
    if not onlyPRI:
	transcols_PRI=transcols_PRI+transcols_DER
    
    if transform:
        for col in transcols_PRI:
            if col in X_all.columns:
                print "log transformation of ",col
                X_all[col]=X_all[col]-X_all[col].min()+1.0
                X_all[col]=X_all[col].apply(np.log)
    
    #correllated=['PRI_jet_subleading_eta','PRI_jet_subleading_phi','PRI_jet_leading_eta','PRI_jet_leading_phi','DER_prodeta_jet_jet','DER_lep_eta_centrality']
    if dropCorrelated:
        #drop correllated features to PRI_jet_subleading_pt
        #drop correllated features to PRI_jet_leading_pt        
        #drop correlated features to DER_deltaeta_jet_jet
        #for col in correllated:
        #    print "Dropping correlated features: ",col
        #    X_all = X_all.drop([col], axis=1)
        X_all=removeCorrelations(X_all,0.995)
    
    if featureFilter is not None:
	print "Using featurefilter..."
	X_all=X_all[featureFilter]
    
    if stats:
        for col in X_all.columns:
            print col
            print X_all[col].describe()
        print X_all.corr()
        #scatter_matrix(X_all, alpha=0.2, figsize=(6, 6), diagonal='hist')
        #plt.show()
	    
	print "idx max observations:" 
	print X_all.apply(lambda x: x.idxmax())
	
	print "idx min observations:"
	print X_all.apply(lambda x: x.idxmin())
    
    
    if plotting:
        #print type(weights)
        #plt.hist(weights[sSelector],bins=50,color='b')
        #plt.hist(weights[bSelector],bins=50,color='r',alpha=0.3)
        X_all.hist()
        plt.show()
        #X[sSelector].hist(color='b', alpha=0.5, bins=50)
        #X[bSelector].hist(color='r', alpha=0.5, bins=50)
    #split data again
    X = X_all[len(X_test.index):]
    X_test = X_all[:len(X_test.index)]
    
    if scale_data:
        X,X_test = scaleData(X,X_test)
    
    if clusterFeature:
	X,y,X_test,weights=clustering(X,y,X_test,weights,n_clusters=4,returnCluster=None,plotting=True)
    
    
    print "Dim train set:",X.shape    
    print "Dim test set :",X_test.shape
    return (X,y,X_test,weights)


def delta_angle_norm(a, b) :
    """
    this and the following functions are taken from kaggle forum
    """
    delta = (a - b)
    delta = delta + (delta < -math.pi) * 2 * math.pi
    delta = delta - (delta > math.pi) * 2 * math.pi
    return delta

def angle_invert(angle, invert_mask) :
    return (
        -999 * (angle == -999) +
        (angle != -999) * (
            angle * (invert_mask == False) +
            (-angle) * (invert_mask == True)))

def reduce_angles(X) :
    """ This function works in-place!"""
    
    delta_angle = 'PRI_tau_phi'
    for angle in ['PRI_lep_phi', 'PRI_met_phi'] :
        X['%s-%s' % (angle, delta_angle)] = delta_angle_norm(X[angle], X[delta_angle])
        del X[angle]
        
    for angle in ['PRI_jet_leading_phi', 'PRI_jet_subleading_phi'] :
        X['%s-%s'% (angle, delta_angle)] = (
            delta_angle_norm(X[angle], X[delta_angle]) * (X[angle] != -999) +
            (-999) * (X[angle] == -999)
        )
        del X[angle]

    del X[delta_angle]
    
    invert_mask = X['PRI_tau_eta'] < 0
    for angle in ['PRI_tau_eta', 'PRI_lep_eta', 'PRI_jet_leading_eta', 'PRI_jet_subleading_eta'] :
        X[angle] = angle_invert(X[angle], invert_mask)
    
    
def AMS(s,b):
    assert s >= 0
    assert b >= 0
    bReg = 10.
    return math.sqrt(2 * ((s + b + bReg) * 
                          math.log(1 + s / (b + bReg)) - s))


def modTrainWeights(wtrain,lytrain,scale_wt=None,verbose=False,normalizeWeights=False,smoothWeights=None):
    """
    Modify training weights
    """
    #verbose=True
    sSelector = lytrain==1
    wsum_s = np.sum(wtrain[sSelector])
    wsum_b = np.sum(wtrain[lytrain==0])
    
    if scale_wt=='auto':             
	scale_wt=wsum_b/wsum_s
    elif scale_wt is None:
	scale_wt=1.0
    wtrain_fit = np.copy(wtrain)  
    wtrain_fit[sSelector] = wtrain[sSelector]*scale_wt
        
    if normalizeWeights:
	wtrain_fit= wtrain_fit*wtrain.shape[0]/np.sum(wtrain_fit)
	print "Normalizing weights, total sum:",np.sum(wtrain_fit)
    
    if smoothWeights is not None:
	print "Smoothing weights by 1/%6.3f"%(smoothWeights)
	wtrain_fit=np.power(wtrain_fit,1/smoothWeights)
	print "w_max: %4.3f w_min: %4.3f mean: %4.3f sdev: %4.3f"%(np.max(wtrain_fit),np.min(wtrain_fit),wtrain_fit.mean(),wtrain_fit.std())
    
    wsum_s_new = np.sum(wtrain_fit[sSelector])
    #wtrain_fit = wtrain*scale_wt
    if verbose:
	plt.hist(wtrain_fit[sSelector],bins=500,color='r',label='s')
	plt.hist(wtrain_fit,bins=500,color='b',alpha=0.3,label='b')
	plt.legend()
	plt.show()

    if verbose: print "Modified train weights: wsum,s: %4.2f wsum,s new: %4.2f ratio,orig: %4.2f ratio,new: %4.2f scale_factor: %8.3f\n"%(wsum_s,wsum_s_new,wsum_b/wsum_s,wsum_b/wsum_s_new,scale_wt)
    return wtrain_fit
  

def amsXvalidation(lmodel,lX,ly,lw,nfolds=5,cutoff=0.5,useProba=True,fitWithWeights=True,useRegressor=False,scale_wt=None,buildModel=True):
    """
    Carries out crossvalidation using AMS metrics
    """
    #ntotal = 250000
    vfunc = np.vectorize(binarizeProbs)
    
    lX = np.asarray(lX)
    ly = np.asarray(ly)
    cv = StratifiedKFold(ly, nfolds)
    #cv = StratifiedShuffleSplit(ly, nfolds, test_size=0.5)
    #cv = ShuffleSplit(ly.shape[0], nfolds, test_size=0.5)
    #cv = KFold(lX.shape[0], n_folds=nfolds,shuffle=True)
    scores=np.zeros(nfolds)
    ams_scores=np.zeros(nfolds)
    scores_train=np.zeros(nfolds)
    ams_scores_train=np.zeros(nfolds)
    for i, (train, test) in enumerate(cv):	
	#train
	lytrain = ly[train]
	wtrain= lw[train]
	if fitWithWeights is False:
	    print "Ignoring weights for fit."
	    lmodel.fit(lX[train],lytrain)
	else:
	 #scale wtrain
         wtrain_fit=modTrainWeights(wtrain,lytrain,scale_wt)
         lmodel.fit(lX[train],lytrain,sample_weight=wtrain_fit)
	
	#training data
	sc_string='AUC'
	if useProba:
	    yinbag=lmodel.predict_proba(lX[train])
	    scores_train[i]=roc_auc_score(lytrain,yinbag[:,1])	   
	else: 
	    yinbag=lmodel.predict(lX[train])
	    if useRegressor:
		  yinbag = vfunc(yinbag,cutoff)
		
	    scores_train[i]=precision_score(lytrain,yinbag)
	    sc_string='PRECISION'

	if cutoff=='compute':
		cutoff = computeCutoff(yinbag[:,1])
	    
	ams_scores_train[i]=ams_score(lytrain,yinbag,sample_weight=wtrain,use_proba=useProba,cutoff=cutoff)
	print "Training %8s=%6.3f AMS=%6.3f" % (sc_string,scores_train[i],ams_scores_train[i])
	
	#test
	truth=ly[test]
	weightsTest=lw[test]
	
	if useProba:
	    yoob=lmodel.predict_proba(lX[test])
	    scores[i]=roc_auc_score(truth,yoob[:,1]) 

	    
	    
	else:
	    yoob=lmodel.predict(lX[test])
	    if useRegressor:
		yoob = vfunc(yoob,cutoff)
	      #scores[i]=f1_score(truth,yoob)
	    scores[i]=precision_score(truth,yoob)

	if cutoff=='compute':
		cutoff = computeCutoff(yoob[:,1])
	    
	ams_scores[i]=ams_score(truth,yoob,sample_weight=weightsTest,use_proba=useProba,cutoff=cutoff)
	print "Iteration=%d %d/%d %-8s=%6.3f AMS=%6.3f\n" % (i+1,train.shape[0], test.shape[0],sc_string,scores[i],ams_scores[i])
	
	
    print "\n##XV SUMMARY##"
    print " <%-8s>: %0.3f (+/- %0.3f)" % (sc_string,scores.mean(), scores.std())
    print " <AMS>: %0.3f (+/- %0.3f)" % (ams_scores.mean(), ams_scores.std())
    print " <%-8s,train>: %0.3f (+/- %0.3f)" % (sc_string,scores_train.mean(), scores_train.std())
    print " <AMS,train>: %0.3f (+/- %0.3f)" % (ams_scores_train.mean(), ams_scores_train.std())
    
    if buildModel:
	print "\n##Building final model##"
	if fitWithWeights:
	    w_fit=modTrainWeights(lw,ly,scale_wt)
	    lmodel.fit(lX,ly,sample_weight=w_fit)
	else:
	    lmodel.fit(lX,ly)
	return model
    else:
	return None

	
def computeCutoff(y_pred,verbose=False):
    """
    computes proba cutoff according to threshhold ratio.
    e.g. threshold_ratio=0.15 keeps the 15% top predictions as signals
    """
    threshold_ratio=0.153
    ntop = int(threshold_ratio * len(y_pred))
    idx_sorted=np.argsort(-y_pred)
    optcutoff=y_pred[idx_sorted][ntop]
    #if verbose: print "ntop: %4d Opt cutoff=%6.3f"%(ntop,optcutoff)
      
    return(optcutoff)

    
    
def ams_score(y_true,y_pred,**kwargs):
    """
    Higgs AMS metric
    """  
    #use scoring weights if available
    opt_cutoff=False
    
    if 'scoring_weight' in kwargs:
	sample_weight=kwargs['scoring_weight']
    elif 'sample_weight' in kwargs:
	sample_weight=kwargs['sample_weight']
    else:
	print "We need sample weights for sensible evaluation!"
	return
	
    verbose=True
    if 'verbose' in kwargs:
	verbose=kwargs['verbose']

    #default parameters
    cutoff=0.5
    ntotal=250000
    
    info="- using 0-1 classification"
    #correcting for shape
    if len(y_pred.shape)>1 and y_pred.shape[1]>1:
	y_pred=y_pred[:,1]
	
    #check if we are dealing with proba 
    if kwargs['use_proba']  and kwargs['cutoff'] is not None:
	if 'cutoff' in kwargs and kwargs['cutoff']=='compute':
	    cutoff = computeCutoff(y_pred,verbose)
	elif 'cutoff' in kwargs :
	    cutoff=kwargs['cutoff']
	info="- using proba. with cutoff=%4.2f"%(cutoff)

       
    if opt_cutoff:
	cutofflist=[0.82,0.83,0.84,0.85,0.86,0.87,0.88]
    else:
	cutofflist=[cutoff]
    
    ams_best=0.0
    cutoff_opt=0.5
    
    for cutoff in cutofflist:
	tpr=0.0
	fpr=0.0 
	for j,row in enumerate(y_pred):
		if row>=cutoff:		
		    if y_true[j]>=cutoff:
			tpr=tpr+sample_weight[j]
		    else:
			fpr=fpr+sample_weight[j]

	sSelector = y_true==1
	bSelector = y_true==0
	#print "Unique weights for signal:",np.unique(sample_weight[sSelector])[0:5]
	
	wfactor=1.* ntotal / y_true.shape[0]
	ssum = np.sum(sample_weight[sSelector])*wfactor
	bsum = np.sum(sample_weight[bSelector])*wfactor
	ams_max=AMS(ssum, 0.0)
	s=tpr*wfactor
	b=fpr*wfactor
	ams=AMS(s,b)
	ams_approx=s/np.power(b,0.5)
	
	if ams>ams_best:
	    ams_best=ams
	    cutoff_opt=cutoff
	if opt_cutoff:
	    print "cutoff=%6.3f AMS= %6.3f"%(cutoff,ams)

    if opt_cutoff:
	print "Optimized cutoff=%6.3f AMS= %6.3f"%(cutoff_opt,ams_best)
	ams = ams_best
	cutoff = cutoff_opt
	
    
    if verbose: 
	wsum=np.sum(y_pred >= cutoff)
	wsum_truth=np.sum(sSelector)
	
	#print '*Sum,weights_s = %8.2f, Sum,weights_b=%8.2f ratio=%8.2f wfactor=%8.2f'%(ssum,bsum,bsum/ssum,wfactor)
	print 'AMS = %6.3f [AMS_max = %6.3f] [AMS,approx = %6.3f] %-32s  ---  ratio(pred): %6.4f ratio(truth): %6.4f wfactor=%8.2f'%(ams,ams_max,ams_approx,info,wsum/(float(y_pred.shape[0])),wsum_truth/(float(y_pred.shape[0])),wfactor)
	
	
	
    return ams   
    
def amsGridsearch(lmodel,lX,ly,lw,fitWithWeights=False,nfolds=5,useProba=False,cutoff=0.5,scale_wt='auto',n_jobs=1,smoothWeights=False):
    print 
    if not 'sample_weight' in inspect.getargspec(lmodel.fit).args:
	  print("WARNING: Fit function ignores sample_weight!")
	  
    fit_params = {'scoring_weight': lw}
    if scale_wt is None:
	fit_params['sample_weight']=lw
    else:
	wtrain_fit=modTrainWeights(lw,ly,scale_wt,smoothWeights=smoothWeights)
	fit_params['sample_weight']=wtrain_fit
	
    fit_params['fitWithWeights']=fitWithWeights
    
    #https://github.com/scikit-learn/scikit-learn/issues/3223 + own modifications
    ams_scorer = make_scorer(score_func=ams_score,use_proba=useProba,needs_proba=useProba,cutoff=cutoff)
    
    #parameters = {'n_estimators':[150,300], 'max_features':[5,10]}#rf
    #parameters = {'n_estimators':[250], 'max_features':[6,8,10],'min_samples_leaf':[5,10]}#xrf+xrf
    #parameters = {'max_depth':[7], 'learning_rate':[0.08],'n_estimators':[100,200,300],'subsample':[0.5],'max_features':[10],'min_samples_leaf':[50]}#gbm
    #parameters = {'max_depth':[7], 'learning_rate':[0.08],'n_estimators':[200],'subsample':[1.0],'max_features':[10],'min_samples_leaf':[20]}#gbm
    parameters = {'max_depth':[6], 'learning_rate':[0.1,0.08,0.05],'n_estimators':[300,500,800],'subsample':[1.0],'loss':['deviance'],'min_samples_leaf':[100],'max_features':[8]}#gbm
    #parameters = {'max_depth':[10], 'learning_rate':[0.001],'n_estimators':[500],'subsample':[0.5],'loss':['deviance']}#gbm
    #parameters = {'max_depth':[15,20,25], 'learning_rate':[0.1,0.01],'n_estimators':[150,300],'subsample':[1.0,0.5]}#gbm
    #parameters = {'max_depth':[20,30], 'learning_rate':[0.1,0.05],'n_estimators':[300,500,1000],'subsample':[0.5],'loss':['exponential']}#gbm
    #parameters = {'max_depth':[15,20], 'learning_rate':[0.05,0.01,0.005],'n_estimators':[250,500],'subsample':[1.0,0.5]}#gbm
    #parameters = {'n_estimators':[100,200,400], 'learning_rate':[0.1,0.05]}#adaboost
    #parameters = {'filter__percentile':[20,15]}#naives bayes
    #parameters = {'filter__percentile': [15], 'model__alpha':[0.0001,0.001],'model__n_iter':[15,50,100],'model__penalty':['l1']}#SGD
    #parameters['model__n_neighbors']=[40,60]}#knn
    #parameters['model__alpha']=[1.0,0.8,0.5,0.1]#opt nb
    #parameters = {'n_neighbors':[10,30,40,50],'algorithm':['ball_tree'],'weights':['distance']}#knn
    clf_opt=grid_search.GridSearchCV(lmodel, parameters,n_jobs=n_jobs,verbose=1,scoring=ams_scorer,cv=nfolds,fit_params=fit_params,refit=True)
    
    clf_opt.fit(lX,ly)
    #dir(clf_opt)
    for params, mean_score, scores in clf_opt.grid_scores_:       
        print("%0.3f (+/- %0.3f) for %r" % (mean_score, scores.std(), params))
    
    scores = cross_validation.cross_val_score(lmodel, lX, ly, fit_params=fit_params,scoring=ams_scorer,cv=nfolds)
    print "AMS: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
    return(clf_opt.best_estimator_)
	
def makePredictions(finalmodel,lXs_test,filename,useProba=True,cutoff=None,printProba=False):
    """
    Uses priorily fit model to make predictions
    """
    print "Preparation of prediction, using test dataframe:",lXs_test.shape
    print finalmodel
    if isinstance(finalmodel,np.ndarray):
	probs = finalmodel
    elif useProba:
	print "Predicting probalities...."
        probs = finalmodel.predict_proba(lXs_test)[:,1]
        #plot it
        plt.hist(probs,label='predictions',bins=50,color='r',alpha=0.3)
        plt.legend()
        plt.show()
        
    else:
        print "Predicting classes...."
	probs = finalmodel.predict(lXs_test)
    
    if not isinstance(cutoff,float):
	cutoff=computeCutoff(probs)
    print "Binarize probabilities with cutoff:",cutoff," and create the labels s+b"
    
    vfunc = np.vectorize(makeLabels)
    labels = vfunc(probs,cutoff)
	
    
    print "Class predictions..."

    print "Rank order..."
    idx_sorted = np.argsort(probs)
    #print idx_sorted
    ro = np.arange(lXs_test.shape[0])+1

    d = {'EventId': lXs_test.index[idx_sorted], 'RankOrder': ro, 'class': labels[idx_sorted]}
    
    print "Saving predictions to: ",filename
    pred_df = pd.DataFrame(data=d)
    #print pred_df
    pred_df.to_csv(filename,index=False) 
    

    
 
def makeLabels(a,cutoff):
    """
    make lables s=1 and b=0
    """
    if a>cutoff: return 's'
    else: return 'b'

    



def analyzeLearningCurve(model,X,y,lw,folds=8):
    """
    make a learning curve according to http://scikit-learn.org/dev/auto_examples/plot_learning_curve.html
    """
    #digits = load_digits()
    #X, y = digits.data, digits.target
    #plt.hist(y,bins=40)
    #cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10, test_size=0.2, random_state=0)
    
    #cv = KFold(X.shape[0], n_folds=folds,shuffle=True)  
    cv = StratifiedKFold(y, n_folds=folds)
    #learn_score = make_scorer(roc_auc_score)
    #learn_score = make_scorer(score_func=ams_score,use_proba=useProba)
    learn_score=make_scorer(f1_score)
    plot_learning_curve(model, "learning curve", X, y, ylim=(0.1, 1.01), cv=cv, n_jobs=4,scoring=learn_score)

    
def buildAMSModel(lmodel,lXs,ly,lw=None,fitWithWeights=True,nfolds=8,useProba=True,cutoff=0.85,scale_wt=None,n_jobs=1,smoothWeights=None,normalizeWeights=False,verbose=False):
    """   
    Final model building part
    """ 
    
    fit_params = {'scoring_weight': lw}
    if scale_wt is None:
	fit_params['sample_weight']=lw
    else:
	fit_params['sample_weight']=modTrainWeights(lw,ly,scale_wt,verbose=verbose,normalizeWeights=normalizeWeights,smoothWeights=smoothWeights)
	
    fit_params['fitWithWeights']=fitWithWeights
    ams_scorer = make_scorer(score_func=ams_score,use_proba=useProba,needs_proba=useProba,cutoff=cutoff)
    
    #how can we avoid that samples are used for fitting
    print "Xvalidation..."
    scores = cross_validation.cross_val_score(lmodel,lXs,ly,fit_params=fit_params, scoring=ams_scorer,cv=nfolds,n_jobs=n_jobs)
    print "<AMS>= %0.4f (+/- %0.4f)" % (scores.mean(), scores.std())
    print "Building model with all instances..."
    
    if fitWithWeights:
	    #just to go sure
	    print "Use sample weights for final model..."
	    final_w=modTrainWeights(lw,ly,scale_wt,normalizeWeights=normalizeWeights,smoothWeights=smoothWeights)
	    lmodel.fit(lXs,ly,sample_weight=final_w)
    else:
	    lmodel.fit(lXs,ly)
    
    #analysis of final predictions
    if useProba:
	print "Using predic_proba for final model..."
	probs = lmodel.predict_proba(lXs)[:,1]
        #plot it
        plt.hist(probs,label='final model',bins=50,color='b')
        plt.legend()
        plt.draw()
    
    
    #analyzeModel(lmodel,feature_names)
    return(lmodel)
    

def checksubmission(filename):
    X = pd.read_csv(filename, sep=",", na_values=['?'], index_col=None)
    print X
    print X.describe()
    
    sums=np.sum(X['class']=='s')
    sumb=np.sum(X['class']=='b')
    print "Signal/Background: %8.5f"%(sums/float(sums+sumb))
  
    print "Unique IDs:",np.unique(X.EventId).shape[0]
    print "Unique ranks:",np.unique(X.RankOrder).shape[0]


def pcAnalysis(X,Xtest,y,w=None,ncomp=2,transform=False):
    """
    PCA 
    """
    
    pca = PCA(n_components=ncomp)
    if transform:
        print "PC reduction"
        X_all = pd.concat([Xtest, X])
        
        X_r = pca.fit_transform(np.asarray(X_all)) 
        print(pca.explained_variance_ratio_)
        #split
        X_r_train = X_r[len(Xtest.index):]
        X_r_test = X_r[:len(Xtest.index)]
        return (X_r_train,X_r_test)
    else:
        print "PC analysis"
        #X_all = pd.concat([Xtest, X])
        X_all = X
        #Uwaga! this is transformation is necessary otherwise PCA gives rubbish!!
        ytrain = np.asarray(y)        
        X_r = pca.fit_transform(np.asarray(X_all))  
        
        if w is None:
            plt.scatter(X_r[ytrain == 0,0], X_r[ytrain == 0,1], c='r', label="background",alpha=0.1)
            plt.scatter(X_r[ytrain == 1,0], X_r[ytrain == 1,1], c='g',label="signal",alpha=0.1)
        else:
            plt.scatter(X_r[ytrain == 0,0], X_r[ytrain == 0,1], c='r', label="background",s=w[ytrain==0]*25.0,alpha=0.1)
            plt.scatter(X_r[ytrain == 1,0], X_r[ytrain == 1,1], c='g',label="signal",s=w[ytrain==1]*1000.0,alpha=0.1)

        print(pca.explained_variance_ratio_) 
        plt.legend()
        #plt.xlim(-3500,2000)
        #plt.ylim(-1000,2000)
        plt.draw()
        
        #clustering        

def clustering(Xtrain,ytrain,Xtest,wtrain=None,n_clusters=3,returnCluster=0,plotting=False):
        """
        Cluster data set
        """
        if returnCluster is not None and returnCluster+1>n_clusters:
            print "Error: returnCluster can not exceed number of clusters!"
                    
        X_all = pd.concat([Xtest, Xtrain])

        centroids,label,inertia = k_means(X_all,n_clusters=n_clusters,verbose=False,n_jobs=2)
        
        pca = PCA(n_components=2)
        X_r = pca.fit_transform(np.asarray(X_all))        
        
        #plt.hist(label,bins=40)
    
        print range(n_clusters)
        cluster_names=['cluster0','cluster1','cluster2','cluster3','cluster4','cluster5']
        for c, i, target_name in zip("rgbkcy", range(n_clusters), cluster_names):
            if plotting:
                plt.scatter(X_r[label == i, 0], X_r[label == i, 1], c=c, label=target_name,alpha=0.1)
            print "Cluster %4d n: %4d"%(i,np.sum(label==i))
        
	train_labels = label[len(Xtest.index):] 
	test_labels = label[:len(Xtest.index)]
        #alternatively we could append cluster label to dataset...                
        #X_new=pd.DataFrame(data=label,columns=['cluster_type'],index=X_all.index)                
     
        #X_sub=pd.concat([X_all, X_new],axis=1)
        
        #Xtrain = X_sub[len(Xtest.index):]
        #Xtest = X_sub[:len(Xtest.index)]
               
        return(train_labels,test_labels)    
        #if returnCluster is not None:                             
            ##print Xtrain['cluster_type']==returnCluster
            #clusterSelect=np.asarray(Xtrain['cluster_type']==returnCluster)                                        
                                            
            #Xtrain_sub = Xtrain.loc[clusterSelect,:]
            ##print type(wtrain)        
            #wtrain_sub = wtrain[clusterSelect]
            #ytrain_sub = ytrain[clusterSelect]
            
            #Xtest_sub = Xtest.loc[Xtest['cluster_type']==returnCluster,:]
            
            #Xtest_sub=Xtest_sub.drop(['cluster_type'],axis=1)       
            #Xtrain_sub=Xtrain_sub.drop(['cluster_type'],axis=1)    
            
            #print "Dim of return train set:",Xtrain_sub.shape  
            #print "Dim of return test set:",Xtest_sub.shape               
            
            ##print Xtrain_sub
            ##print wtrain_sub
            ##print ytrain_sub        
            #return (Xtrain_sub,ytrain_sub,Xtest_sub,wtrain_sub)
        #else:    
            #return (Xtrain,ytrain,Xtest,wtrain)

            
def divideAndConquer(model,Xtrain,ytrain,Xtest,wtrain=None,n_clusters=3):
    """
    splits data, builds model and then combines everything
    """ 
    train_labels, test_labels = clustering(Xtrain,ytrain,Xtest,wtrain,n_clusters=3,returnCluster=0,plotting=False)
    
    for i in range(n_clusters):
	idx_train = train_labels == i
	idx_test = test_labels == i
	
	lXtrain = Xtrain.iloc[idx_train]
	lytrain = ytrain[idx_train]
	lwtrain = wtrain[idx_train]
	
	lXtest = Xtest.iloc[idx_test]
	
	#print lXtrain.describe()
	print "Dim X,before:",lXtrain.shape
	print "Dim Xtest,before:",lXtest.shape
	lXtrain,lXtest = removeZeroVariance(lXtrain,lXtest)
	print "Dim X:",lXtrain.shape
	print "Dim Xtest:",lXtest.shape
	
	#need to rescale weights
	model=buildAMSModel(model,lXtrain,lytrain,lwtrain,nfolds=4,fitWithWeights=fitWithWeights,useProba=useProba,cutoff=cutoff,scale_wt=scale_wt,n_jobs=1) 
	
	
    
    
if __name__=="__main__":
    """
    Main classTrue
    """
    #sample weights
    #http://scikit-learn.org/stable/auto_examples/svm/plot_weighted_samples.html
    #sample weights
    #https://github.com/scikit-learn/scikit-learn/pull/3224
    # Set a seed for consistant results
    # TODO make only class prediction and guess ranking... 
    # TODO http://www.rdkit.org/docs/Cookbook.html parallel stuff
    # TODO http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer
    # TODO http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html
    # TODO scale sample weights????http://pastebin.com/VcL3dWiK
    # TODO predict weights???
    # weights as np.array
    # group by jet
    # group by different NA
    #we should optimise precision instead of recall
    # sample_weights are small for signals? look at TPR vs. FPR wenn weights are scaled for signals 
    # we need training weights and scoring weights
    t0 = time()
    
    print "numpy:",np.__version__
    print "pandas:",pd.__version__
    print "sklearn:",sl.__version__
    #pd.set_option('display.height', 5)
    pd.set_option('display.max_columns', 14)
    pd.set_option('display.max_rows', 40)
    
    np.random.seed(1234)
    nsamples=-1
    onlyPRI='' #'PRI' or 'DER'
    #createNAFeats='DER_mass_MMC_NA' #brings something?
    createNAFeats=None
    dropCorrelated=False
    dropFeatures=None #[u'PRI_jet_subleading_eta',u'PRI_jet_subleading_phi','PRI_jet_num']
    scale_data=False #bringt nichts by NB
    replaceNA=False
    plotting=False
    stats=False
    transform=False
    useProba=True  #use probailities for prediction
    fitWithWeights=True #use weights for training
    scale_wt='auto'
    #scale_wt=None
    #scale_wt=5000
    useRegressor=False
    #cutoff='compute'
    cutoff='compute'
    clusterFeature=False
    #polyFeatures=['PRI_tau_ptXPRI_met','PRI_tau_etaXPRI_lep_eta','PRI_lep_ptXPRI_met','PRI_lep_phiXPRI_met_phi','PRI_tau_ptXPRI_lep_pt']
    #polyFeatures=['PRI_tau_ptXPRI_met','PRI_tau_etaXPRI_lep_eta']
    polyFeatures=None
    #quadraticFeatures=['PRI_lep_eta_^2','PRI_tau_eta_^2']
    quadraticFeatures=None
    imputeMassModel=None
    #imputeMassModel=RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='mse', max_features=5,oob_score=False)#LinearRegression()##SGDRegressor(alpha=0.0001,n_iter=5,shuffle=False,loss='squared_loss',penalty='l2')#GaussianNB()#KNeighborsClassifier(n_neighbors=100)#
    createMassEstimate=True
    smoothWeights=1.0
    normalizeWeights=False
    verbose=False
    subfile="/home/loschen/Desktop/datamining-kaggle/higgs/submissions/sub0909c.csv"
    featureFilter=None
    loadCake=False
    #onehotenc=[u'PRI_jet_num']
    onehotenc=None
    #featureFilter=[u'DER_mass_MMC', u'DER_mass_transverse_met_lep', u'DER_mass_vis', u'metXp_lep_vec', u'PRI_tau_pt', u'DER_met_phi_centrality', u'DER_pt_ratio_lep_tau', u'DER_deltar_tau_lep', u'p_lepXp_tau_vec', u'metXp_tau_vec', u'PRI_met', u'metXp_lep', u'DER_sum_pt', u'DER_pt_tot', u'DER_pt_h', u'PRI_lep_phi-PRI_tau_phi', u'PRI_lep_eta', u'DER_deltaeta_jet_jet', u'PRI_met_sumet', u'PRI_lep_pt', u'PRI_jet_leading_eta', u'metXp_tau', u'p_tauXp_lep', u'p_lep_abs', u'PRI_met_phi-PRI_tau_phi', u'PRI_tau_eta', u'p_tau_abs', u'DER_mass_jet_jet', u'PRI_lep_phi', u'PRI_met_phi', u'DER_lep_eta_centrality', u'PRI_jet_all_pt', u'PRI_jet_leading_pt', u'PRI_jet_leading_phi-PRI_tau_phi', u'DER_prodeta_jet_jet', u'PRI_jet_subleading_eta', u'PRI_jet_num', u'PRI_jet_subleading_pt']#ordered after RF importance AMS= 3.51 GBM,AMS=3.6290 
    #featureFilter=[u'DER_mass_MMC', u'DER_mass_transverse_met_lep', u'DER_mass_vis', u'metXp_lep_vec', u'PRI_tau_pt', u'DER_met_phi_centrality', u'DER_pt_ratio_lep_tau', u'DER_deltar_tau_lep', u'p_lepXp_tau_vec', u'metXp_tau_vec', u'PRI_met', u'metXp_lep', u'DER_sum_pt', u'DER_pt_tot', u'DER_pt_h', u'PRI_lep_phi-PRI_tau_phi', u'PRI_lep_eta', u'DER_deltaeta_jet_jet', u'PRI_met_sumet', u'PRI_lep_pt', u'PRI_jet_leading_eta', u'metXp_tau', u'p_tauXp_lep', u'p_lep_abs', u'PRI_met_phi-PRI_tau_phi', u'PRI_tau_eta', u'p_tau_abs', u'DER_mass_jet_jet', u'PRI_lep_phi', u'PRI_met_phi', u'DER_lep_eta_centrality', u'PRI_jet_all_pt', u'PRI_jet_leading_pt', u'PRI_jet_leading_phi-PRI_tau_phi', u'DER_prodeta_jet_jet']
    Xtrain,ytrain,Xtest,wtrain=prepareDatasets(nsamples,onlyPRI,replaceNA,plotting,stats,transform,createNAFeats,dropCorrelated,scale_data,clusterFeature,dropFeatures,polyFeatures,createMassEstimate,imputeMassModel,featureFilter,onehotenc,quadraticFeatures,loadCake)
    #nfolds=8#
    nfolds=StratifiedShuffleSplit(ytrain, n_iter=16, test_size=0.5)
    #nfolds=ShuffleSplit(ytrain.shape[0], n_iter=8, test_size=0.25)
    #nfolds = StratifiedKFold(ytrain, 8)
    #nfolds = KFold(ytrain.shape[0], 4,shuffle=True)
    #pcAnalysis(Xtrain,Xtest,ytrain,wtrain,ncomp=2,transform=False)       
    #RF cluster1 AMS=2.600 (77544)
    #RF cluster2 AMS=4.331 (72543)
    #RF cluster3 AMS=3.742 (99913 samples) ~ weighted average AMS = 3.56
    #NB cluster1 AMS=1.850 (77544 samples)
    #NB cluster2 AMS=1.834 (72543 samples)
    #NB cluster3 AMS=2.467 (9950,100,200,913 samples)
    #NB all AMS=1.162
    #NB all AMS=1.315 (variable transformation)
    #NB all AMS=1.315 (transformation+proba=0.5)
    #NB all AMS=1.374 (transformation+proba=0.15)
    #NB all AMS=1.473 (transformation+proba=0.05)
    #NB all AMS=1.641 (transformation+proba=0.01)
    #NB all AMS=1.745 (transformation+proba=0.01,replaceNA)
    #NB all AMS=1.822 (transformation+proba=0.15,replaceNA,dropcorrelated)
    #print Xtrain.describe()
    #model = RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto')#AMS~1.85
    #model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, class_weight='auto')
    #model = SGDClassifier(alpha=0.1,n_iter=150,shuffle=True,loss='log',penalty='l1',n_jobs=4)#SW,correct implementation?
    #model = GaussianNB()#AMS~1.2
    #model = GradientBoostingClassifier(loss='exponential',n_estimators=150,learning_rate=.2,max_depth=6,verbose=2,subsample=1.0)   
    #model = GradientBoostingClassifier(loss='deviance', learning_rate=0.001, n_estimators=500, subsample=1.0, max_depth=10, max_features='auto',init=None,verbose=False)#opt fitting wo weights!!!
    #model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=150, subsample=0.5, max_depth=25, max_features='auto',init=None,verbose=False)#opt2
    
    #model = GradientBoostingRegressor(n_estimators=150,learning_rate=.1,max_depth=6,verbose=1,subsample=1.0)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=150, learning_rate=0.05,max_depth=6,min_samples_leaf=100,max_features='auto',verbose=0)#with no weights, cutoff=0.85 and useProba=True
    #model = pyGridSearch(model,Xtrain,ytrain)
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=15)), ('model', GaussianNB())])
    #model = KNeighborsClassifier(n_neighbors=5,weights='distance',algorithm='ball_tree')#AMS~2.245
    print Xtrain.columns
    #model = KNeighborsClassifier(n_neighbors=10)
    #model = AdaBoostClassifier(n_estimators=200,learning_rate=0.1)
    #model=GaussianNB()
    #model = SVC(C=1.0,gamma=0.0)
    
    #model = ExtraTreesClassifier(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=1,criterion='entropy', max_features=10,oob_score=False)##scale_wt 600 cutoff 0.85
    #model = AdaBoostClassifier(n_estimators=100,learning_rate=0.1)    
    #analyzeLearningCurve(model,Xtrain,ytrain,wtrain)
    #odel = ExtraTreesClassifier(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=4,criterion='entropy', max_features=5,oob_score=False)#opt
    #model = Pipeline([('filter', SelectPercentile(f_classif, percentile=15)), ('model', model)])
    #model = amsGridsearch(model,Xtrain,yt    #smoothWeights=Falserain,wtrain,fitWithWeights=fitWithWeights,nfolds=nfolds,useProba=useProba,cutoff=cutoff)
    model = GradientBoostingClassifier(loss='deviance',n_estimators=150, learning_rate=0.1, max_depth=6,subsample=1.0,verbose=False) #opt weight =500 AMS=3.548
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=500, learning_rate=0.05, max_depth=6,subsample=1.0,max_features=8,min_samples_leaf=100,verbose=0) #opt weight =500 AMS=3.548
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=800, learning_rate=0.02, max_depth=6,subsample=.5,max_features=8,min_samples_leaf=100,verbose=False) 
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=2000, learning_rate=0.01, max_depth=7,subsample=0.5,max_features=10,min_samples_leaf=50,verbose=0)#3.72  | 3.69
    #model = XgboostClassifier(n_estimators=120,learning_rate=0.1,max_depth=6,n_jobs=1,NA=-999.0)
    #model =  RandomForestClassifier(n_estimators=250,max_depth=None,min_samples_leaf=5,n_jobs=1,criterion='entropy', max_features=5,oob_score=False)#SW-proba=False ams=3.42
    #model = AdaBoostClassifier(base_estimator=basemodel,n_estimators=10,learning_rate=0.5)   
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=200, learning_rate=0.08, max_depth=7,subsample=1.0,max_features=10,min_samples_leaf=20,verbose=False)
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.08, max_depth=6,subsample=1.0,max_features=10,min_samples_leaf=50,verbose=False)#AMS=3.678
    #model = GradientBoostingClassifier(loss='deviance',n_estimators=500, learning_rate=0.05, max_depth=6,subsample=1.0,max_features=6,min_samples_leaf=100,verbose=False)#new opt
    #model = BaggingClassifier(base_estimator=basemodel,n_estimators=10,n_jobs=1,verbose=1)
    model=buildAMSModel(model,Xtrain,ytrain,wtrain,nfolds=nfolds,fitWithWeights=fitWithWeights,useProba=useProba,cutoff=cutoff,scale_wt=scale_wt,n_jobs=8,smoothWeights=smoothWeights,normalizeWeights=normalizeWeights,verbose=verbose) 
    #divideAndConquer(model,Xtrain,ytrain,Xtest,wtrain,n_clusters=3)#did not work
    #model = amsXvalidation(model,Xtrain,ytrain,wtrain,nfolds=nfolds,cutoff=cutoff,useProba=useProba,fitWithWeights=fitWithWeights,useRegressor=useRegressor,scale_wt=scale_wt,buildModel=True)    
    #iterativeFeatureSelection(model,Xtrain,Xtest,ytrain,1,1)    
    #model= amsXvalidation(model,Xtrain,ytrain,wtrain,nfolds=nfolds,cutoff=cutoff,useProba=useProba,fitWithWeights=fitWithWeights,useRegressor=useRegressor,scale_wt=scale_wt,buildModel=True)
    #clist=[0.25,0.50,0.75,0.85]
    #for c in clist:
	#  cutoff=c
	 # print "c",c
	  #model=amsXvalidation(model,Xtrain,ytrain,wtrain,nfolds=nfolds,cutoff=cutoff,useProba=useProba,fitWithWeights=fitWithWeights,useRegressor=useRegressor,scale_wt=scale_wt,buildModel=False)
	  #model=buildAMSModel(model,Xtrain,ytrain,wtrain,nfolds=nfolds,fitWithWeights=fitWithWeights,useProba=useProba,cutoff=cutoff,scale_wt=scale_wt)
    #model = amsGridsearch(model,Xtrain,ytrain,wtrain,fitWithWeights=fitWithWeights,nfolds=nfolds,useProba=useProba,smoothWeights=smoothWeights,cutoff=cutoff,scale_wt=scale_wt,n_jobs=8)
    print model
    #makePredictions(model,Xtest,subfile,useProba=useProba,cutoff=cutoff)
    #checksubmission(subfile)
    print("Model building done on %d samples in %fs" % (Xtrain.shape[0],time() - t0))
    plt.show()