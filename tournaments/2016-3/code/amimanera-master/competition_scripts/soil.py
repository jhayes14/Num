#!/usr/bin/python 
# coding: utf-8
# Code source: Chrissly31415
# License: BSD

import pandas as pd
import numpy as np
import sklearn as sl
import scipy as sp
from sklearn.base import clone
from qsprLib import *
import random
from savitzky_golay import *
from sklearn import manifold


def prepareDatasets(nsamples=-1,onlySpectra=False,deleteSpectra=False,plotting=False,standardize=True,doPCA=None,findPeaks=None,makeDerivative=None,featureFilter=None,loadFeatures=None,deleteFeatures=None,removeVar=0.0,removeCor=None,useSavitzkyGolay=False,addNoiseColumns=None,addLandscapes=False,compressIR=500,transform=None):
    Xtrain = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/training.csv')
    Xtest = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/sorted_test.csv')
    ymat = Xtrain[['Ca','P','pH','SOC','Sand']]
    Xtrain.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
    
    if nsamples != -1: 
	rows = random.sample(Xtrain.index, nsamples)
	Xtrain = Xtrain.ix[rows,:]
	ymat = ymat.ix[rows,:]
      
    Xtest.drop('PIDN', axis=1, inplace=True)
    
    #combine data
    X_orig = pd.concat([Xtest, Xtrain],ignore_index=True)   
    X_orig['Depth'] = X_orig['Depth'].replace(['Topsoil','Subsoil'],[1,0])
    X_all = X_orig.copy()
    #X_all = pd.concat([X_orig, pd.get_dummies(X_orig['Depth'])],axis=1)
    #X_all.drop(['Depth','Subsoil'], axis=1, inplace=True)
    #X_all.rename(columns = {'Topsoil':'Depth'})
    
    if findPeaks is not None and not False:
      if findPeaks is 'load':
	  X_ir = peakfinder('load')
      else:
	  X_ir = peakfinder(X_all)
      print "X_ir",X_ir['int3720.0'].values
      X_all = pd.concat([X_all, X_ir],axis=1)
      print "da shape",X_all.shape
      #remove zero columns
      #X_all = X_all.ix[:,(X_all != 0).any(axis=0)]
    
    if transform is not None:
	print "log transformation "
	if "spectra" == transform:
	    transform = [m for m in list(X_all.columns) if m[0]=='m']
	#spectra = [m for m in list(X_all.columns) if m[0]=='m']
        for col in X_all[transform]:
            if col in X_all.columns:
                X_all[col]=X_all[col]-X_all[col].min()+1.0
                #X_all[col]=X_all[col]-X_all[col].min()+1.0
                X_all[col]=X_all[col].apply(np.log)
	
	#X_all.loc[:,transform].hist(bins=30)
        #plt.show()
    
    if useSavitzkyGolay:
	spectra = [m for m in list(X_all.columns) if m[0]=='m']
	X_SP = applySavitzkyGolay(X_all[spectra])
	X_all.drop(spectra, axis=1, inplace=True) 
	X_all = pd.concat([X_all, X_SP],axis=1)   
    
    if makeDerivative is not None:
	X_diff = makeDiff(X_all.iloc[:,:3578])
	if '2nd' in makeDerivative:
	    X_diff = makeDiff(X_diff)
	X_all = pd.concat([X_all, X_diff],axis=1)
	X_all = X_all.ix[:,(X_all != 0).any(axis=0)]
	deleteSpectra=True
    
    if compressIR is not None:
	start_str='m'
	if useSavitzkyGolay: start_str='s'
	if makeDerivative: start_str='d'
	spectra = [m for m in list(X_all.columns) if m[0]==start_str]
	X_cpr = makeCompression(X_all[spectra],compressIR)
	X_all.drop(spectra, axis=1, inplace=True) 
	X_all = pd.concat([X_all, X_cpr],axis=1)
    
    
	
    
    if onlySpectra:
	X_all = X_all.iloc[:,:3578]

    if deleteSpectra:
	spectra = [m for m in list(X_all.columns) if m[0]=='m']
	X_all.drop(spectra, axis=1, inplace=True) 
    
    if featureFilter is not None:
	print "Using featurefilter..."
	X_all=X_all[featureFilter]
    
    if doPCA is not None:
	pca = PCA(n_components=doPCA)
	X_all = pd.DataFrame(pca.fit_transform(np.asarray(X_all)))
	for col in X_all.columns:
	    X_all=X_all.rename(columns = {col:'pca'+str(col+1)})
	
	print "explained variance:",pca.explained_variance_ratio_
	print "components: %5d sum: %6.2f"%(doPCA,pca.explained_variance_ratio_.sum())
      
    if deleteFeatures is not None:
	deleteFeatures = [ col for col in deleteFeatures if col in X_all.columns ]
	X_all.drop(deleteFeatures, axis=1, inplace=True) 
    
    if addLandscapes is True:
	X_all = modelLandscapes(X_all,X_orig)
    
    if addNoiseColumns is not None:
	Xrnd = pd.DataFrame(np.random.randn(X_all.shape[0],addNoiseColumns))
	#print "Xrnd:",Xrnd.shape
	#print Xrnd
	for col in Xrnd.columns:
	    Xrnd=Xrnd.rename(columns = {col:'rnd'+str(col+1)})
	
	X_all = pd.concat([X_all, Xrnd],axis=1)
    
    if removeCor is not None:
	X_all = removeCorrelations(X_all,removeCor)

   
    if standardize:
	X_all = scaleData(X_all)
	
    X_all = removeLowVariance(X_all,removeVar)
    
    #split data again
    Xtrain = X_all[len(Xtest.index):]
    Xtest = X_all[:len(Xtest.index)]
    #print Xtrain.describe()
    #print Xtest.describe()
    
    #analyze data
    #if plotting:
	#axs = Xtrain.hist(bins=30,color='b',alpha=0.3)
	#for ax, (colname, values) in zip(axs.flat, Xtest.iteritems()):
	    #values.hist(ax=ax, bins=30,color='g',alpha=0.3)
	#plt.show()
    
    if loadFeatures is not None:
	tmp1 = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/training_all.csv')[loadFeatures]
	tmp1.index = Xtrain.index
	Xtrain = pd.concat([Xtrain,tmp1],axis=1)
	tmp2 = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/test_all.csv')[loadFeatures]
	tmp2.index = Xtest.index
	print tmp2.head(10)
	print Xtest.head(10)
	Xtest = pd.concat([Xtest,tmp2],axis=1)
   
    #analyze data
    if plotting:
	Xtrain.hist(bins=30,label='legend')
	#Xtest.hist(bins=50)
      
	#somehow ordering is wrong???
	#axs2 = Xtrain.hist(bins=30)
	#for ax2, (colname, values) in zip(axs2.flat, Xtest.iteritems()):
	#    values.hist(ax=ax2,alpha=0.3, bins=30)
	plt.legend()
	plt.show()
    
    #print "Train set:\n",Xtrain.describe()
    #print "Test set:\n",Xtest.describe()
    #print Xtest.index
    print "Dim train set:",Xtrain.shape    
    print "Dim test set :",Xtest.shape
    
    #Xtrain.to_csv("/home/loschen/Desktop/datamining-kaggle/african_soil/training_all.csv",index=False)
    #Xtest.to_csv("/home/loschen/Desktop/datamining-kaggle/african_soil/test_all.csv",index=False)
    
    return(Xtrain,Xtest,ymat)

def applySavitzkyGolay(X,window_size=31, order=3,plotting=False):
    """
    use filter
    """
    print "Use Savitzky-Golay (windowsize=%4d, order=%4d)"%(window_size,order)
    tutto=[]
    for ind in X.index:
	row=[]
	row.append(ind)
	yorig = X.ix[ind,:].values
	ysg = savitzky_golay(yorig, window_size=41, order=3)
	
	for el in ysg:
	    row.append(el)
	tutto.append(row)
	#
	if plotting:
	    plt.plot(yorig, label='Noisy signal')
	    plt.plot( ysg, 'r', label='Filtered signal')
	    plt.legend(loc="best")
	    plt.show()
	
    newdf = pd.DataFrame(tutto).set_index(0)
    colnames = [ "s"+str(x) for x in xrange(newdf.shape[1]) ]
    newdf.columns=colnames
    print newdf.head(10)
    return(newdf)
    

def makeCompression(X,bin_size,plotting=False):
    """
    Collects spectra frequencies
    """
    print "Compressing spectrum"
    tutto=[]
    
    for ind in X.index:
	row=[]
	row.append(ind)
	data = X.ix[ind,:].values	
	start_bins = X.ix[ind,:].values.shape[0]
	base = np.linspace(0,start_bins ,start_bins)
	bins = np.linspace(0,start_bins ,bin_size+1)

	bin_means1 = np.histogram(base, bins=bins, weights=data)[0]
	tmp = np.histogram(base, bins=bin_size)[0]
	bin_means1= bin_means1 / tmp
	for el in bin_means1:
	    row.append(el)
	tutto.append(row)

    newdf = pd.DataFrame(tutto).set_index(0)
    colnames = [ "z"+str(x) for x in xrange(newdf.shape[1]) ]
    newdf.columns=colnames
    if plotting:
	newdf.iloc[3,:].plot()
	plt.show()
    
    print newdf.head(10)
    ###print newdf.describe()
    return(newdf)
	
	
    
    
def makeDiff(X,plotting=False):
    """
    make derivative
    """
    print "Making 1st derivative..."
    tutto=[]
    for ind in X.index:
	row=[]
	row.append(ind)
	for el in np.gradient(X.ix[ind,:].values):
	    row.append(el)
	tutto.append(row)
    newdf = pd.DataFrame(tutto).set_index(0)
    colnames = [ "d"+str(x) for x in xrange(newdf.shape[1]) ]
    newdf.columns=colnames
    if plotting:
	newdf.iloc[3,:].plot()
	plt.show()
    
    print newdf.head(10)
    ###print newdf.describe()
    return(newdf)
    
def peakfinder(X_all,verbose=False):
    """
    Finds peak in spectrum, and creates dataframe from it
    """
    from scipy import signal
    #xs = np.arange(0, 4*np.pi, 0.05)
    #data = np.sin(xs)
    
    print "Locating IR-peaks..."
    if X_all is "load":
	newdf = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/irpeaks.csv')
	return(newdf)
    
    print X_all.describe()
    tutto=[]
    print X_all.index
    print X_all.index[-1]
    for ind in X_all.index:
	print "processing spectra %4d/%4d "%(ind,X_all.shape[0])
	row=[]
	row.append(ind)
	data = X_all.ix[ind,:3578].values.flatten()
	peakind = signal.find_peaks_cwt(data, widths=np.arange(5,25),max_distances=np.arange(5,25)/5,noise_perc=10,min_snr=1,min_length=1,gap_thresh=5)
	(hi,edges) = np.histogram(peakind,bins=100,range=(0,4000))
	edges=edges[0:-1]
	for el in hi:
	    row.append(el)
	tutto.append(row)
	
	if verbose:
	    print hi
	    plt.plot(data)
	    plt.bar(edges,hi>0)
	    plt.show()
	    
    newdf = pd.DataFrame(tutto).set_index(0)
    colnames = [ "int"+str(x) for x in edges.tolist() ]
    newdf.columns=colnames   
    #print newdf.head(10)
    #print newdf.describe()
    newdf.to_csv("/home/loschen/Desktop/datamining-kaggle/african_soil/irpeaks.csv",index=False)
    return(newdf)
    
    

    

def gridSearchModels(lmodels,lX,lymat,fit_params=None,scoring='mean_squared_error',cv_split=8,n_jobs=8):
    """
    Do grid search on several models
    """
    pass
    
    
def makePrediction(models,Xtrain,Xtest,nt,filename='subXXX.csv'):
    preds = np.zeros((Xtest.shape[0], nt))
    f, axarr = plt.subplots(nt, sharex=True)
    rmse_list = np.zeros((nt,1))
    print "%-10s %6s" %("TARGET","RMSE,train")
    for i in range(nt):
	y_true = np.asarray(ymat.iloc[:,i])
	y_pred = models[i].predict(Xtrain).astype(float)
	rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))
	print "%-10s %6.3f" %(ymat.columns[i],rmse_score)
	preds[:,i] = models[i].predict(Xtest).astype(float)
	axarr[i].scatter(y_pred,y_true)
	axarr[i].set_ylabel(ymat.columns[i])
	rmse_list[i]=rmse_score
    
    print "<RMSE,train>: %6.3f +/- %6.3f" %(rmse_list.mean(),rmse_list.std())
    plt.show()
    sample = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/sample_submission.csv')
    sample['Ca'] = preds[:,0]
    sample['P'] = preds[:,1]
    sample['pH'] = preds[:,2]
    sample['SOC'] = preds[:,3]
    sample['Sand'] = preds[:,4]
    sample.to_csv(filename, index = False)    

def modelsFeatureSelection(lmodels,Xold,Xold_test,lymat):
    for i,model in enumerate(lmodels):
	iterativeFeatureSelection(model,Xold,Xold_test,lymat.iloc[:,i],1,1)

def modelsGreedySelection(lmodels,Xold,Xold_test,lymat):
    for i,model in enumerate(lmodels):
	if i!=1: continue
	print "Target:",ymat.columns[i]
	greedyFeatureSelection(models[0],Xtrain,ymat.iloc[:,i],itermax=60,itermin=30,targets=None,start_features=None,n_jobs=8,verbose=False,cv= cross_validation.LeavePLabelOut(pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/landscapes_quick3.csv',index_col=0)['LANDSCAPE'],1))
	
def modelLandscapes(X_all,X_orig):
    plottLS=True
    groups = pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/groupings.csv',sep=';')
    groups['TMAP']=np.round(groups['TMAP'],5)
    print X_all.index
    print X_orig.index
    X_all['TMAP']=np.round(X_orig['TMAP'],5)
    
    #print X_all.loc[:,['TMAP']]
    #merge mixes train and test
    X_all = X_all.reset_index().merge( groups, how="left",on='TMAP')
    #get original order
    X_all.sort('index',inplace=True)
    X_all = X_all.set_index('index')
    
    #X_all.drop(['TMAP'], axis=1, inplace=True)
    X_all['LANDSCAPE']=X_all['LANDSCAPE'].str.lstrip('LS').astype(int)
    X_all[['LANDSCAPE']].reset_index(drop=True).to_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/landscapes_new.csv')
    #print X_all.loc[:,['TMAP','LANDSCAPE']]
    
    
    #X_all.drop(['LANDSCAPE'], axis=1, inplace=True)
    #print X_all.index

    #return X_all

    #X_all = X_all[X_all.LANDSCAPE != 10]
    #X_all['LIMESTONE']= np.bitwise_or(np.bitwise_or(X_all.LANDSCAPE == 'LS10',X_all.LANDSCAPE == 'LS39'),X_all.LANDSCAPE == 'LS40').astype(int)
    #print X_all['LIMESTONE'].describe()
    
    #one hot encoding not good 
    #X_all = pd.concat([X_all, pd.get_dummies(X_all['LANDSCAPE'])],axis=1)
    #X_all.drop(['LANDSCAPE'], axis=1, inplace=True)
    
    #grouping by landscpae only spectra
    X_spectra = X_all.iloc[:,:3578]
    X_spectra['LANDSCAPE'] = X_all['LANDSCAPE']
    
    gb = X_spectra.groupby('LANDSCAPE',sort=False)
    X_LS = gb.aggregate(np.mean)
    
    #PCA only 
    #pca = PCA(n_components=2)
    #X_r = pd.DataFrame(pca.fit_transform(np.asarray(X_LS)),columns=['LSPCA1','LSPCA2'])
    #X_r.index = X_LS.index
       
    mds = manifold.MDS(n_components=2, dissimilarity="euclidean", random_state=6)
    X_r = pd.DataFrame(mds.fit_transform(np.asarray(X_LS)),columns=['MS1','MS2'])
    X_r.index = X_LS.index
    
    
    if plottLS:
	#X_LS.iloc[0:20,:].T.plot()
	print X_LS.describe()
	print X_LS.head(20)
	plt.scatter(X_r.iloc[:,0].values, X_r.iloc[:,1].values, c='r', label="landscape",alpha=0.5)
	for name, x, y in zip(X_r.index, X_r.iloc[:,0].values, X_r.iloc[:,1].values):
	    plt.annotate(name,xy = (x, y))
	plt.show()

    X_r['LANDSCAPE']=X_r.index
    #X_r.to_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/groupings_pca.csv')
    #same issue again, indices are shuffeld upon merge
    X_all = X_all.reset_index().merge(X_r,how="left", on='LANDSCAPE')
    X_all.sort('index',inplace=True)
    X_all = X_all.set_index('index')
    
    X_all.drop(['LANDSCAPE'], axis=1, inplace=True)
    print X_all
    return X_all
    
def getFeatures(key):
    """
    Collections of some use features
    """
    #all vars
    features={}
    # optimzed R leaps features
    features['Ca']=["m7328.26","m7067.91","m7056.34","m6265.66","m5237.78","m4751.8","m4406.6","m3853.12","m3851.19","m3642.92","m3627.49","m2917.8","m2873.45","m2850.31","m2603.46","m2547.53","m2537.89","m2505.11","m1859.06","m1853.28","m1833.99","m1801.21","m1795.42","m1793.49","m1791.57","m1787.71","m1781.92","m1776.14","m1770.35","m1610.29","m1608.36","m1529.29","m1448.3","m1313.3","m1267.02","m1230.38","m1228.45","m1191.81","m1135.88","m1076.1","m784.895","m779.109","m723.183","m713.541","m711.612","m647.972","m628.687","m599.76","LSTN","REF2"]
    features['P']=["m7494.11","m7353.33","m7254.97","m7235.69","m7173.98","m6265.66","m4528.09","m4524.23","m3752.84","m3704.63","m2917.8","m2873.45","m2850.31","m2547.53","m2537.89","m2514.75","m2360.47","m2335.4","m2065.41","m2061.55","m1735.64","m1712.5","m1687.43","m1610.29","m1513.86","m1506.15","m1448.3","m1442.51","m1425.15","m1232.3","m1230.38","m1228.45","m1106.95","m1105.02","m1079.95","m1078.03","m1074.17","m1064.53","m952.673","m877.462","m873.605","m852.392","m815.751","m813.822","m777.181","m678.828","BSAN","EVI","REF7","RELI"]
    features['pH']=["m7177.83","m7170.12","m4836.65","m4406.6","m3851.19","m3534.92","m3515.63","m3394.14","m2850.31","m2489.68","m2238.98","m2215.83","m2159.91","m2065.41","m1758.78","m1735.64","m1726","m1700.93","m1685.5","m1664.29","m1652.71","m1623.79","m1618","m1583.29","m1575.58","m1438.65","m1338.37","m1294.02","m1270.87","m1130.09","m1079.95","m1072.24","m1024.03","m1008.6","m954.602","m917.961","m908.318","m835.036","m821.536","m788.752","m617.116","m599.76","BSAN","BSAV","ELEV","EVI","LSTD","LSTN","RELI","TMAP"]
    features['SOC']=["m7177.83","m7067.91","m5237.78","m3733.55","m3687.27","m2514.75","m2238.98","m2217.76","m2042.27","m1959.34","m1830.14","m1814.71","m1780","m1756.85","m1716.35","m1710.57","m1687.43","m1673.93","m1627.64","m1621.86","m1585.22","m1562.08","m1544.72","m1542.79","m1519.65","m1496.51","m1477.22","m1243.88","m1191.81","m1164.81","m1162.88","m1160.95","m1114.67","m1110.81","m1097.31","m1054.88","m1022.1","m917.961","m906.39","m894.819","m892.89","m890.962","m869.748","m767.539","m750.182","m701.97","m647.972","REF1","REF2","RELI"]
    features['Sand']=["m7490.25","m7459.39","m7347.54","m6265.66","m4836.65","m4751.8","m4510.74","m4196.39","m3658.34","m2159.91","m1922.7","m1920.77","m1841.71","m1772.28","m1687.43","m1685.5","m1637.29","m1592.93","m1488.79","m1394.3","m1326.8","m1324.87","m1191.81","m1164.81","m1162.88","m1160.95","m1157.09","m1155.16","m1130.09","m1097.31","m1078.03","m1037.53","m983.529","m952.673","m850.464","m844.678","m806.108","m802.251","m796.466","m784.895","m779.109","m769.467","m767.539","m723.183","BSAV","ELEV","LSTD","REF1","REF7","TMAP"]
    features['GA_Ca']=["m7497.96","m7496.04","m7494.11","m7492.18","m7490.25","m7486.39","m7469.04","m7434.32","m7424.68","m7374.54","m7353.33","m7351.4","m7347.54","m7314.76","m7256.9","m7235.69","m7177.83","m7173.98","m7133.48","m7069.84","m7062.13","m7058.27","m6265.66","m5237.78","m4836.65","m4751.8","m4535.81","m4533.88","m4531.95","m4528.09","m4524.23","m4522.31","m4516.52","m4510.74","m4483.74","m4406.6","m4196.39","m3851.19","m3750.91","m3739.34","m3737.41","m3735.48","m3733.55","m3729.7","m3725.84","m3723.91","m3720.05","m3716.2","m3714.27","m3710.41","m3704.63","m3702.7","m3700.77","m3696.91","m3689.2","m3685.34","m3683.41","m3681.48","m3677.63","m3675.7","m3673.77","m3671.84","m3669.91","m3666.06","m3662.2","m3658.34","m3656.41","m3650.63","m3646.77","m3644.84","m3642.92","m3639.06","m3637.13","m3633.27","m3629.42","m3625.56","m3619.77","m3617.84","m3612.06","m3610.13","m3608.2","m3606.27","m3546.49","m3544.56","m3525.28","m3523.35","m3515.63","m3451.99","m2917.8","m2603.46","m2547.53","m2532.11",\
"m2530.18","m2528.25","m2514.75","m2503.18","m2501.25","m2364.33","m2360.47","m2350.83","m2346.97","m2343.11","m2341.19","m2333.47","m2312.26","m2219.69","m2210.05","m2057.7","m2053.84","m2051.91","m2048.05","m2046.13","m2044.2","m2042.27","m2040.34","m2032.63","m2028.77","m2026.84","m2022.98","m2021.06","m2009.49","m2007.56","m1963.2","m1922.7","m1918.85","m1914.99","m1909.2","m1905.35","m1893.78","m1891.85","m1864.85","m1857.13","m1855.21","m1851.35","m1849.42","m1845.56","m1843.64","m1837.85","m1835.92","m1833.99","m1828.21","m1822.42","m1820.49","m1818.56","m1812.78","m1808.92","m1806.99","m1805.07","m1801.21","m1795.42","m1785.78","m1783.85","m1781.92","m1774.21","m1770.35","m1758.78","m1754.92","m1753","m1749.14","m1747.21","m1745.28","m1743.35","m1741.43","m1739.5","m1735.64","m1733.71","m1729.85","m1722.14","m1720.21","m1714.43","m1712.5","m1702.86","m1700.93","m1691.28","m1689.36","m1681.64","m1679.71","m1677.79","m1673.93","m1672","m1670.07","m1668.14","m1666.21","m1643.07","m1637.29","m1635.36",\
"m1623.79","m1621.86","m1619.93","m1612.22","m1606.43","m1602.57","m1600.65","m1594.86","m1585.22","m1581.36","m1579.43","m1567.86","m1565.93","m1562.08","m1560.15","m1558.22","m1556.29","m1552.43","m1548.58","m1546.65","m1544.72","m1540.86","m1537.01","m1531.22","m1529.29","m1527.36","m1521.58","m1519.65","m1513.86","m1508.08","m1504.22","m1498.44","m1488.79","m1479.15","m1477.22","m1475.29","m1456.01","m1446.37","m1442.51","m1438.65","m1436.72","m1429.01","m1427.08","m1421.3","m1415.51","m1413.58","m1400.08","m1394.3","m1382.73","m1378.87","m1376.94","m1375.01","m1373.08","m1334.51","m1332.59","m1324.87","m1321.01","m1319.09","m1317.16","m1315.23","m1309.44","m1307.52","m1299.8","m1290.16","m1278.59","m1276.66","m1268.95","m1267.02","m1261.23","m1259.3","m1255.45","m1251.59","m1247.73","m1245.8","m1243.88","m1241.95","m1240.02","m1236.16","m1232.3","m1228.45","m1164.81","m1160.95","m1157.09","m1135.88","m1132.02","m1130.09","m1128.17","m1122.38","m1118.52","m1112.74","m1110.81","m1105.02","m1103.1",\
"m1076.1"\
,"m1074.17","m1070.31","m1064.53","m1062.6","m1060.67","m1058.74","m1047.17","m1045.24","m1043.31","m1037.53","m1029.81","m1020.17","m1014.39","m1010.53","m998.957","m997.029","m991.243","m989.315","m983.529","m979.672","m971.958","m968.101","m966.173","m962.316","m960.387","m956.53","m954.602","m952.673","m923.746","m919.889","m916.032","m914.104","m906.39","m904.461","m881.319","m879.391","m877.462","m875.534","m871.677","m846.607","m842.75","m840.821","m838.893","m836.964","m827.322","m825.393","m819.608","m815.751","m813.822","m809.965","m804.18","m800.323","m798.394","m788.752","m786.823","m784.895","m781.038","m777.181","m775.252","m773.324","m771.395","m767.539","m759.825","m750.182","m723.183","m719.326","m717.398","m715.469","m713.541","m709.684","m703.898","m686.542","m680.757","m678.828","m667.257","m655.686","m649.901","m647.972","m646.044","m644.115","m636.401","m634.473","m628.687","m626.759","m624.83","m622.902","m619.045","m609.402","m607.474","m599.76","BSAN","BSAS","BSAV","EVI","LSTD"\
,"LSTN","REF1","REF3","REF7","RELI","TMFI"]


    features['Ca_greedy']=['z49', 'z61', 'z123', 'RELI', 'Depth', 'z110', 'z108', 'z109', 'z51', 'z111', 'z129', 'z140', 'z144', 'z9', 'z117', 'z106', 'z133', 'z145', 'z52', 'z47', 'z95', 'z92', 'z99', 'ELEV', 'z114', 'z122', 'z136', 'z148', 'z90', 'z94', 'z68', 'z147', 'LSTN', 'z96', 'z89', 'z91', 'REF1', 'z104', 'z138', 'TMAP', 'REF3']
    features['P_greedy']=['z2', 'z5', 'z119', 'z130', 'z139', 'z74', 'z135', 'Depth', 'z95', 'z125']
    features['pH_greedy']=['TMAP', 'z133', 'z130', 'z82', 'z134', 'z127', 'z99', 'z115', 'z110', 'z0', 'ELEV', 'z131', 'z70', 'z47', 'RELI', 'z141', 'z146', 'z116', 'z114', 'z128', 'z101', 'z120', 'z143', 'z145', 'Depth', 'CTI', 'REF2', 'z83', 'z17', 'z136', 'z71', 'z118', 'z67', 'z113']
    #geht viellt. besser
    features['SOC_greedy']=['z100', 'z81', 'z136', 'z117', 'z114', 'RELI', 'REF2', 'z122', 'z128', 'z99', 'z131', 'z147', 'z60', 'z64', 'z138', 'LSTD', 'z137', 'z97', 'z22', 'CTI', 'z143', 'z141', 'z139', 'Depth', 'BSAN', 'LSTN', 'z68', 'z7', 'z13', 'z0', 'z6','z5']
    features['Sand_greedy']=['z137', 'z133', 'TMAP', 'z83', 'z0', 'z138', 'z134', 'z142', 'z120', 'z74', 'z122', 'z144', 'z145', 'z140', 'z139', 'z146', 'z135', 'z117', 'z87', 'z141', 'CTI', 'Depth', 'z103', 'z109', 'z121', 'RELI', 'z136', 'z143', 'z48', 'z9', 'z93', 'z115']
    
    features['Ca_test']=['z49', 'z61', 'z123', 'RELI', 'Depth', 'z110', 'z108', 'z109', 'z51', 'z111', 'z129', 'z140', 'z144', 'z9', 'z117', 'z106', 'z133', 'z145', 'z52', 'z47', 'z95', 'z92', 'z99', 'ELEV', 'z114', 'z122', 'z136', 'z148', 'z90', 'z94', 'z68', 'z147']
    features['P_test']=['z2', 'z5', 'z119', 'z130', 'z139', 'z74', 'z135', 'Depth', 'z95', 'z125']
    features['pH_test']=['TMAP', 'z133', 'z130', 'z82', 'z134', 'z127', 'z99', 'z115', 'z110', 'z0', 'ELEV', 'z131', 'z70', 'z47', 'RELI', 'z141', 'z146', 'z116', 'z114', 'z128', 'z101', 'z120', 'z143', 'z145', 'Depth', 'CTI', 'REF2', 'z83', 'z17', 'z136', 'z71', 'z118']
    features['SOC_test']=['z100', 'z81', 'z136', 'z117', 'z114', 'RELI', 'REF2', 'z122', 'z128', 'z99', 'z131', 'z147', 'z60', 'z64', 'z138', 'LSTD', 'z137', 'z97', 'z22', 'CTI', 'z143', 'z141', 'z139', 'Depth', 'BSAN', 'LSTN', 'z68', 'z7', 'z13', 'z0', 'z6', 'z5']
    features['Sand_test']=['z137', 'z133', 'TMAP', 'z83', 'z0', 'z138', 'z134', 'z142', 'z120', 'z74', 'z122', 'z144', 'z145', 'z140', 'z139', 'z146', 'z135', 'z117', 'z87', 'z141', 'CTI', 'Depth', 'z103', 'z109', 'z121', 'RELI', 'z136', 'z143', 'z48', 'z9', 'z93', 'z115']
    
    #greedy selection with svm and 1st derivatives
    features['Ca_svm']=['z144', 'z131', 'z125', 'z124', 'z115', 'z138', 'z102', 'z145', 'z128', 'z49']
    features['P_svm']=['z3', 'z116', 'ELEV', 'z113', 'z130', 'z42']
    features['pH_svm']=['z112', 'z132', 'z130', 'z147', 'z111', 'Depth', 'z49', 'z95', 'z126', 'z31', 'z96', 'z116', 'z131', 'z87', 'z72', 'z124']
    features['SOC_svm']=['z124', 'z122', 'z129', 'z130', 'z52', 'z55', 'z53', 'z117', 'z59', 'z126', 'z140', 'z131', 'z1', 'REF7', 'z134', 'z4', 'z58']
    features['Sand_svm']=['z113', 'z136', 'z82', 'z141', 'z140', 'z9', 'z121', 'z95', 'z24', 'z114', 'z135', 'Depth', 'z127', 'z23', 'z21', 'z19', 'z70', 'z18']
    
    features['non-spectra']=['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
    features['non-spectra+depth']=['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI','DEPTH']
    features['co2']= ['m2379.76', 'm2377.83', 'm2375.9', 'm2373.97','m2372.04', 'm2370.11', 'm2368.18', 'm2366.26','m2364.33', 'm2362.4', 'm2360.47', 'm2358.54','m2356.61', 'm2354.68', 'm2352.76']
    #only spectrad
    return features[key]

    
def buildmodels(lmodels,lX,lymat,fit_params=None,scoring='mean_squared_error',cv_split=8,n_jobs=8,gridSearch=False,useLandscapeCV=True):
    
    if useLandscapeCV:
	#split across landscapes
	#cv = cross_validation.LeavePLabelOut(pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/landscapes_int.csv',index_col=0)['LANDSCAPE'],1)#37
	#cv = cross_validation.LeavePLabelOut(pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/landscapes_fast.csv',index_col=0)['LANDSCAPE'],1)#19
	#cv = cross_validation.LeavePLabelOut(pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/landscapes_quick.csv',index_col=0)['LANDSCAPE'],1)#10
	#cv = cross_validation.LeavePLabelOut(pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/landscapes_quick2.csv',index_col=0)['LANDSCAPE'],1)#9
	cv = cross_validation.LeavePLabelOut(pd.read_csv('/home/loschen/Desktop/datamining-kaggle/african_soil/landscapes_quick3.csv',index_col=0)['LANDSCAPE'],1)#8
	#for train_index, test_index in cv:
	#    #print("TRAIN:", train_index, "TEST:", test_index)
	#    print "dim train:",train_index.shape
	#    print "dim test:",test_index.shape
	#    print "len(cv)",len(cv)
	#    print lX.iloc[train_index].head(65)
	#    print lX.iloc[test_index].head(65)
	    
	print "Leave one landscape out x-validation, %4d groups."%(len(cv))
    else:
	cv = cross_validation.ShuffleSplit(lX.shape[0], n_iter=cv_split, test_size=0.25, random_state=0)
    
    scores=np.zeros((lymat.shape[1],len(cv)))
    for i in range(lymat.shape[1]):
	ly = lymat.iloc[:,i].values
	#be carefull sign is flipped
	if gridSearch is True:
	    #parameters = {'filter__param': [100]}#normal pipeline    
	    #parameters = {'filter__k': [2000,3000,3594],'pca__n_components':[0.99,0.995], 'model__alpha': [10000,100,1.0,0.01,0.0001] }#pipeline
	    #parameters = {'filter__k': [2000,3000,3594],'pca__n_components':[0.99], 'model__alpha': [10000,100,1.0,0.01,0.0001] }#pipeline
	    parameters = {'filter__param': [99],'model__gamma':np.logspace(-4, -1, 6), 'model__C': [10,100,1000,10000],'model__epsilon':[0.1,0.01,0.001,0.0001] }#SVR
	    #parameters = {'filter__param': [99],'model__alpha': [0.001],'model__loss':['huber'],'model__penalty':['elasticnet'],'model__epsilon':[1.0],'model__n_iter':[200]}#pipeline
	    #parameters = {'varfilter__threshold': [0.0,0.1,0.001,0.0001,0.00001] }#pipeline
	    #parameters = {'filter__k': [10,20],'pca__n_components':[10,20], 'model__alpha': [1.0] }#pipeline
	    #parameters = {'pca__n_components':[20,25,30,40,50,60],'filter__param': [98]}#PCR regression
	    #parameters = {'pca__n_components':[35,40,100,200,300,400,500]}
	    #parameters = {'filter__param': [100,98,80,70,50],'model__alpha':[1000,100,10,1.0]}#ridge
	    #parameters = {'pca__n_components':[100,150,200],'model__alpha':[100,10,0.1]}#ridge
	    #parameters = {'model__n_neighbors':[5,10,25]}#KNN
	    #parameters = {'filter__param': [98,80],'model__n_components':[5,20,40]}#PLS
	    #parameters = {'filter__param': [99],'model__alpha':[0.01,0.001],'model__l1_ratio':[0.001,0.0005]}#elastic net
	    #parameters = {'filter__param': [100,98,90,80,50],'model__alpha':[10,1,0.1,0.01,0.001]}#RIDGECV
	    #parameters = {'filter__param': [100,99],'model__n_neighbors':[3,4]}#KNN
	    #parameters = {'model__max_depth':[5,6], 'model__learning_rate':[0.1],'model__n_estimators':[200,300,400],'model__subsample':[1.0],'model__loss':['huber'],'model__min_samples_leaf':[10],'model__max_features':[None]}
	    #parameters = {'filter__param': [100],'model__loss': ['huber'],'model__n_estimators':[150,500,1000]}#GBR
	    #parameters = {'n_estimators':[250], 'max_features':['auto'],'min_samples_leaf':[1]}#xrf+xrf
	    #parameters = {'filter__param': [40,20,5],'model__n_estimators':[200]}#RF
	    clf  = grid_search.GridSearchCV(lmodels[i], parameters,n_jobs=n_jobs,verbose=0,scoring=scoring,cv=cv,fit_params=fit_params,refit=True)
	    clf.fit(lX,ly)
	    best_score=1.0E5
	    print("%6s %6s %6s %r" % ("OOB", "MEAN", "SDEV", "PARAMS"))
	    for params, mean_score, cvscores in clf.grid_scores_:
		oob_score = (-1*mean_score)**0.5
		cvscores = (-1*cvscores)**0.5
		mean_score = cvscores.mean()
		print("%6.3f %6.3f %6.3f %r" % (oob_score, mean_score, cvscores.std(), params))
		if mean_score < best_score:
		    best_score = mean_score
		    scores[i,:] = cvscores
		
	    lmodels[i] = clf.best_estimator_
	     
	else:    
	    scores[i,:] = (-1*cross_validation.cross_val_score(lmodels[i],lX,ly,fit_params=fit_params, scoring=scoring,cv=cv,n_jobs=n_jobs))**0.5   

	print "TARGET: %-12s"%(lymat.columns[i]),    
	print " - <score>= %0.3f (+/- %0.3f) runs: %4d" % (scores[i].mean(), scores[i].std(),scores.shape[1])
	#FIT FULL MODEL
	lmodels[i].fit(lX,ly)

    print 
    print "Total cv-score: %6.3f (+/- %6.3f) "%(scores.mean(axis=1).mean(),scores.mean(axis=1).std())
    return(models)
    
    
if __name__=="__main__":
    #TODO https://gist.github.com/sixtenbe/1178136
    #TODO http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html
    #TODO 2nd derivative using np.diff twice, using np.gradient
    #TODO Savitzky-Golay or something selfmade?
    #TODO continuum removal
    #TODO PLS
    #TODO outlier detection for P
    #TODO RFECV
    #TODO polynomial on extra features
    #TODO make TMAP categorical/binary
    #TODO python string to integer:df[0] = df[0].str.replace(r'[$,]', '').astype('float')
    #http://stackoverflow.com/questions/3172509/numpy-convert-categorical-string-arrays-to-an-integer-array
    #http://stackoverflow.com/questions/15356433/how-to-generate-pandas-dataframe-column-of-categorical-from-string-column
    #http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    #boruta results: 
  
    t0 = time()
    print "numpy:",np.__version__
    print "scipy:",sp.__version__
    print "pandas:",pd.__version__
    print "sklearn:",sl.__version__
    pd.set_option('display.max_columns', 14)
    pd.set_option('display.max_rows', 40)

    np.random.seed(123)
    nsamples=-1
    onlySpectra=False
    plotting=False
    standardize=True
    doPCA=None
    findPeaks=None
    #findPeaks='load'
    makeDerivative='1st'#CHECK indices after derivative making...
    featureFilter=None#getFeatures('Ca_svm')+getFeatures('non-spectra')
    removeVar=0.1
    useSavitzkyGolay=False
    deleteSpectra=useSavitzkyGolay
    addNoiseColumns=None
    addLandscapes=False
    compressIR=300

    loadFeatures=None#getFeatures('non-spectra')
    transform=None#'spectra'#['RELI','CTI'] #['RELI','RELI','REF7','CTI','BSAS','BSAV']
    deleteFeatures=getFeatures('co2')
    removeCor=None
    
    (Xtrain,Xtest,ymat) = prepareDatasets(nsamples,onlySpectra,deleteSpectra,plotting,standardize,doPCA,findPeaks,makeDerivative,featureFilter,loadFeatures,deleteFeatures,removeVar,removeCor,useSavitzkyGolay,addNoiseColumns,addLandscapes,compressIR,transform)
    
    #pcAnalysis(Xtrain,Xtest)alphas=[0.1]
    #print Xtrain.columns
    #print Xtest.columns
    #ymat = ymat.iloc[:,1:2]
    nt = ymat.shape[1]

    #generate models
    #C:Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    models=[]
    for i in range(nt):
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', PLSRegression(n_components=30))])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', LassoLarsCV())])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', LinearRegression())])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', ElasticNet(alpha=.01,l1_ratio=0.001,max_iter=1000))])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', RidgeCV(alphas=[1.0]))])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', Ridge(alpha=1.0))])
	#model = Pipeline([('model', KNeighborsRegressor(n_neighbors=5))])
	#model = GaussianNB()
	#model = Pipeline([('pca', PCA(n_components=doPCA)),('model', Ridge())])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', Ridge(alpha=0.1))])
	#model = Pipeline([('pca', PCA(n_components=doPCA)),('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', LinearRegression())])
	#model = Pipeline([('pca', PCA(n_components=doPCA)),('model', RidgeCV())])
	#model = RidgeCV(alphas=[ 0.05,0.1])
	#model = SGDRegressor(alpha=0.1,n_iter=50,shuffle=True,loss='squared_loss',penalty='l1')#too many features
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SGDRegressor(alpha=0.001,n_iter=300,shuffle=True,loss='huber',epsilon=1.0,penalty='l2'))])
	#model = Pipeline([('pca', PCA(n_components=doPCA)), ('model', LinearRegression())])
	#model = Pipeline([('pca', PCA(n_components=doPCA)), ('model', SGDRegressor(alpha=0.00001,n_iter=150,shuffle=True,loss='squared_loss',penalty='l2'))])
	#model = Pipeline([('pca', PCA(n_components=200)), ('model', RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_split=2,min_samples_leaf=5,n_jobs=1,criterion='mse', max_features='auto',oob_score=False))])
	#model = Pipeline([('svd', TruncatedSVD(n_components=25, algorithm='randomized', n_iter=5, tol=0.0)), ('model', RandomForestRegressor(n_estimators=250,max_depth=None,min_samples_split=2,min_samples_leaf=5,n_jobs=1,criterion='mse', max_features='auto',oob_score=False))])
	#model = Pipeline([('pca', PCA(n_components=doPCA)), ('model', SVR(C=1.0, gamma=0.0, verbose = 0))])
	#model = Pipeline([('filter', SelectPercentile(f_regression, percentile=50)), ('model', SVR(kernel='rbf',epsilon=0.1,C=10000.0, gamma=0.0, verbose = 0))])
	#model = Pipeline([('filter',SelectPercentile(f_regression, percentile=99)), ('model', SGDRegressor(alpha=0.001,n_iter=250,shuffle=True,loss='huber',penalty='elasticnet',epsilon=1.0))])
	#model = Pipeline([('filter',SelectPercentile(f_regression, pe6,8,10rcentile=99)), ('model', BayesianRidge())])
	#model = Pipeline([('filter',GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model',GradientBoostingRegressor(loss='huber',n_estimators=150, learning_rate=0.1, max_depth=2,subsample=1.0))])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model',KNeighborsRegressor(n_neighbors=5, weights='uniform') )])
	#model = Pipeline([('filter', SelectKBest(f_regression, k=10)),('pca', PCA(n_components=10)), ('model', Ridge())])
	
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', SVR(C=10000.0, gamma=0.0, verbose = 0))])
	model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(kernel='rbf',epsilon=0.1,C=100.0, gamma=0.0005, verbose = 0))])
	
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', Lasso())])
	#model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', LarsCV())])#grottig
	#model = SVR(C=10000.0, gamma=0.0, verbose = 0)
	#model = SVR(C=10000.0, gamma=0.0005, verbose = 0)
	#model = RandomForestRegressor(n_estimators=500,max_depth=None,min_samples_split=2,min_samples_leaf=5,n_jobs=1,criterion='mse', max_features='auto',oob_score=False)
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', RandomForestRegressor(n_estimators=500))])
	#model = LinearRegression(normalize=False)
	#model = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', BaggingRegressor(base_estimator=PLSRegression(n_components=30),n_estimators=10,n_jobs=1,verbose=0))])
	models.append(model) 
    #individual model SVR
    #models[0] =  Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=100.0, gamma=0.005, verbose = 0))])#Ca RMSE=0.287
    #models[1] =  Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', SVR(C=100000.0, gamma=0.0005, verbose = 0))]) #P RMSE=0.886
    #models[2] =  Pipeline([('filter', GenericUnivariateSelect(f_regression, param=95,mode='percentile')), ('model', SVR(C=10000.0, gamma=0.0005, verbose = 0))])#pH RMSE=0.321
    #models[3] =  Pipeline([('filter', GenericUnivariateSelect(f_regression, param=90,mode='percentile')), ('model', SVR(C=10000.0, gamma=0.0005, verbose = 0))])#SOC RMSE=0.278
    #models[4] =  Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', SVR(C=10000.0, gam'model', ElasticNet(alpha=.01,l1_ratio=0.001,max_iter=1000)ma=0.0005, verbose = 0))])#Sand RMSE=0.316
    
    #individual model PLS
    #models[0] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', PLSRegression(n_components=20))])#Ca RMSE=0.384
    #models[1] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', PLSRegression(n_components=20))])#P RMSE=0.886
    #models[2] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', PLSRegression(n_components=50))])#pH RMSE=0.346
    #models[3] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', PLSRegression(n_components=40))])#SOC RMSE =0.348
    #models[4] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', PLSRegression(n_components=40))])#Sand RMSE=0.356
    
    #individual model PLS LOO-CV
    #models[0] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', PLSRegression(n_components=20))])#Ca RMSE=0.384 (+/- 0.214)
    #models[1] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=80,mode='percentile')), ('model', PLSRegression(n_components=5))])#P RMSE=0.896 (+/- 0.396)
    #models[2] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=100,mode='percentile')), ('model', PLSRegression(n_components=50))])#pH RMSE=0.473 (+/- 0.074)
    #models[3] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=90,mode='percentile')), ('model', PLSRegression(n_components=25))])#SOC RMSE =0.522 (+/- 0.278)
    #models[4] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=98,mode='percentile')), ('model', PLSRegression(n_components=20))])#Sand RMSE=0.487 (+/- 0.099)
    
    #individual model elastic net LOO-CV
    #models[0] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')),('model', ElasticNet(alpha=.01,l1_ratio=0.001,max_iter=1000)) ])#Ca RMSE=0.301 (+/- 0.287) 
    #models[1] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')),('model', ElasticNet(alpha=.01,l1_ratio=0.001,max_iter=1000)) ])#P RMSE=0.754 (+/- 0.643)
    #models[2] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')),('model', ElasticNet(alpha=.01,l1_ratio=0.001,max_iter=1000))])#pH RMSE=0.453 (+/- 0.166)
    #models[3] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', ElasticNet(alpha=.001,l1_ratio=0.001,max_iter=1000))])#SOC RMSE =0.461 (+/- 0.362)
    #models[4] = Pipeline([('filter', GenericUnivariateSelect(f_regression, param=99,mode='percentile')), ('model', ElasticNet(alpha=.001,l1_ratio=0.001,max_iter=1000))])#Sand RMSE=0.445 (+/- 0.229) 
    
    
    #make the training
    models = buildmodels(models,Xtrain,ymat,cv_split=8,gridSearch=True,n_jobs=8,useLandscapeCV=True)
    #showMisclass(models[0],Xtrain,ymat.iloc[:,0],t=2.0)
    #modelsFeatureSelection(models,Xtrain,Xtest,ymat)
    #modelsGreedySelection(models,Xtrain,Xtest,ymat)
    
    for i in range(nt):
	print "TARGET: %-10s" %(ymat.columns[i])
	#print models[i].alpha_#optimized alpha from ridge 0.1 0.05
	print models[i]

      
    #makePrediction(models,Xtrain,Xtest,nt,filename='/home/loschen/Desktop/datamining-kaggle/african_soil/submissions/sub2609a.csv')
    #make the predictions 

    print("Model building done on %d samples in %fs" % (Xtrain.shape[0],time() - t0))

