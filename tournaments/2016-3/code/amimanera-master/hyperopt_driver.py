#!/usr/bin/python 
# coding: utf-8
#https://github.com/hyperopt/hyperopt/wiki/FMin
#http://fastml.com/optimizing-hyperparams-with-hyperopt/
#https://github.com/zygmuntz/kaggle-burn-cpu/blob/master/driver.py
#https://github.com/hyperopt/hyperopt-sklearn

from hyperopt import fmin, tpe, hp
from qsprLib import *
from cater import *
import math

space_xgb = ( 
	#hp.loguniform( 'learning_rate', np.log(0.01), np.log(0.2) ),
	hp.uniform( 'learning_rate',0.01 ,0.05),
	hp.quniform( 'max_depth', 7, 15,1 ),
	hp.uniform( 'subsample', 0.5, 1.0 ),
	hp.uniform( 'colsample_bytree', 0.5, 1.0 ),
	#hp.uniform( 'max_features', 0.9, 1.0 ),
	#hp.uniform( 'max_samples', 0.9, 1.0 ),
	#hp.choice( 'bootstrap', [ False] ),
	hp.uniform( 'gamma', 0.1, 3.0 ),
	hp.quniform( 'min_child_weight', 1, 10,1 )
)

space_nn  = ( 
	hp.quniform( 'hidden1_num_units', 400,600,50),
	hp.quniform( 'hidden2_num_units', 400,600,50 ),
	hp.quniform( 'hidden3_num_units', 250,600,50 ),
	#hp.uniform( 'dropout0_p', 0.0, 0.0),
	hp.uniform( 'dropout1_p', 0.0, 0.25 ),
	hp.uniform( 'dropout2_p', 0.0, 0.25 ),
	hp.uniform( 'dropout3_p', 0.0, 0.25 ),
	hp.quniform( 'max_epochs', 50,150,25),
	hp.loguniform( 'learning_rate', np.log( 1E-4 ), np.log( 1E-2 )),
	hp.loguniform( 'L2_alpha', np.log( 1E-6 ), np.log( 1E-2 )),
	#hp.uniform( 'leakiness', 0.1, 0.3 ),
	#hp.uniform( 'max_features', 0.9, 1.0 ),
	#hp.uniform( 'max_samples', 0.9, 1.0 )
	
)


def func_nn(params):
      global counter
      global X
      global ta
      
      counter += 1
      print "Iteration:        %d"%(counter)
      s = time()
      
      #hidden1_num_units,hidden2_num_units,hidden3_num_units,dropout0_p,dropout1_p,dropout2_p,dropout3_p,max_epochs,learning_rate,L2_alpha = params
      #dropout0_p,dropout1_p,dropout2_p,dropout3_p,max_epochs,learning_rate,L2_alpha = params
      #hidden1_num_units,hidden2_num_units,hidden3_num_units,dropout0_p,max_epochs,learning_rate,L2_alpha,max_features = params
      #hidden1_num_units,hidden2_num_units,dropout0_p,dropout1_p,dropout2_p,max_epochs,learning_rate,max_features = params
      #hidden1_num_units,hidden2_num_units,hidden3_num_units,dropout1_p,dropout2_p,dropout3_p,max_epochs,learning_rate,L2_alpha,max_features,max_samples = params
      hidden1_num_units,hidden2_num_units,hidden3_num_units,dropout1_p,dropout2_p,dropout3_p,max_epochs,learning_rate,L2_alpha = params
      dropout0_p=0.0
      #dropout2_p=0.25
      #dropout3_p=0.25
      
      print "hidden1_num_units:    %6d"% (hidden1_num_units)
      print "hidden2_num_units:    %6d"% (hidden2_num_units)
      print "hidden3_num_units:    %6d"% (hidden3_num_units)
      #print "dropout0_p:          %6.2f"%(dropout0_p)
      print "dropout1_p:          %6.2f"%(dropout1_p)
      print "dropout2_p:          %6.2f"%(dropout2_p)
      print "dropout3_p:          %6.2f"%(dropout3_p)
      print "max_epochs:          %6d"% (max_epochs)
      print "learning_rate:       %6.2e"%(learning_rate)
      print "L2_alpha:            %6.2e"%(L2_alpha)
      #print "leakiness:            %6.2e"%(leakiness)
      input_shape =X.shape[1]
      #input_shape = int(math.floor(X.shape[1]*max_features))
      #print "max_features: 	 %6.4f (%6d)"%(max_features,input_shape)
      #print "max_samples: 	 %6.4f"%(max_samples)
      
      print input_shape
      #input_shape = 271

      
      basemodel = NeuralNet(layers=[('input', layers.InputLayer),
	#('dropout0', layers.DropoutLayer),
	('hidden1', layers.DenseLayer),
	('dropout1', layers.DropoutLayer),
	('hidden2', layers.DenseLayer),
	('dropout2', layers.DropoutLayer),
	('hidden3', layers.DenseLayer),
	('dropout3', layers.DropoutLayer),
	('output', layers.DenseLayer)],

	input_shape=(None, input_shape),
	#dropout0_p=dropout0_p,

	hidden1_num_units=hidden1_num_units,
	hidden1_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout1_p=dropout1_p,

	hidden2_num_units=hidden2_num_units,
	hidden2_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout2_p=dropout2_p,
	
	hidden3_num_units=hidden3_num_units,
	hidden3_nonlinearity=nonlinearities.LeakyRectify(leakiness=0.1),
	dropout3_p=dropout3_p,

	output_num_units=1,
	output_nonlinearity=None,

	regression=True,
	objective=RMSE,
	objective_alpha=L2_alpha,
	batch_iterator_train=ShuffleBatchIterator(batch_size = 64),

	#update=adagrad,#0.001
	update=rmsprop,
	update_learning_rate=theano.shared(float32(learning_rate)),
	
	eval_size=0.0,
	verbose=1,
	max_epochs=max_epochs,

	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=learning_rate, stop=0.00005),
		#EarlyStopping(patience=20),
		],
	)

      
      model = basemodel
      #model = BaggingRegressor(base_estimator=basemodel,n_estimators=3,n_jobs=1,verbose=2,random_state=None,max_samples=max_samples,max_features=max_features,bootstrap=False)
      score = buildModel(model,X,y,cv=KLabelFolds(pd.Series(ta), n_folds=5, repeats =1),scoring=scoring_func,n_jobs=1,trainFull=False,verbose=True)
      #score =  buildXvalModel(model,X,y,refit=False,cv=KLabelFolds(pd.Series(ta), n_folds=5, repeats =1))

      
      print ">>score: %6.3f (+/- %6.3f)"%(-1*score.mean(),score.std())
      print "elapsed: {}s \n".format( int( round( time() - s )))
      return -1*score.mean()
	
def func_xgb(params):
      global counter
      global ta
      
      counter += 1
      print "Iteration:        %d"%(counter)
      s = time()
      
      learning_rate, max_depth, subsample,colsample_bytree,gamma,min_child_weight = params
      #learning_rate, max_depth, subsample,colsample_bytree,max_features,max_samples,bootstrap = params
      print "learning_rate:    %6.4f"% (learning_rate)
      print "max_depth:        %6.4f" %(max_depth)
      print "subsample:        %6.4f"%(subsample)
      print "colsample_bytree: %6.4f"%(colsample_bytree)
      print "gamma: 		 %6.4f"%(gamma)
      print "min_child_weight: %6.4f"%(min_child_weight)
      #print "max_features: 	 %6.4f"%(max_features)
      #print "max_samples: 	 %6.4f"%(max_samples)
      #print "bootstrap: 	 %6d"%(bootstrap)
      
      #model = XgboostRegressor(n_estimators=400,learning_rate=learning_rate,max_depth=max_depth,subsample=subsample,colsample_bytree=colsample_bytree,min_child_weight=5,n_jobs=2,objective='reg:linear',eval_metric='rmse',booster='gbtree',eval_size=0.0,silent=1)
      model = XgboostRegressor(n_estimators=4000,learning_rate=learning_rate,max_depth=max_depth,subsample=subsample,colsample_bytree=colsample_bytree,min_child_weight=5,n_jobs=1,objective='reg:linear',eval_metric='rmse',booster='gbtree',eval_size=0.0,silent=1)
      #model = BaggingRegressor(base_estimator=model,n_estimators=3,n_jobs=1,verbose=0,random_state=None,max_samples=max_samples,max_features=max_features,bootstrap=bootstrap)
      score = buildModel(model,X,y,cv=KLabelFolds(pd.Series(ta), n_folds=8, repeats =1),scoring=scoring_func,n_jobs=8,trainFull=False,verbose=True)
      
      #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.15, random_state=42)
      #model.fit( Xtrain, ytrain )
      #p = model.predict_proba( Xtest )

      #score = multiclass_log_loss(ytest, p)
      print ">>score: %6.3f (+/- %6.3f)"%(-1*score.mean(),score.std()),
      print " elapsed: {}s \n".format( int( round( time() - s )))
      return -1*score.mean()


"""
scoring_func = make_scorer(multiclass_log_loss, greater_is_better=False, needs_proba=True)
counter=0
#(X,y,Xtest,labels) = prepareDataset(nsamples='shuffle',addFeatures=True)
all_features=[u'feat_1', u'feat_2', u'feat_3', u'feat_4', u'feat_5', u'feat_6', u'feat_7', u'feat_8', u'feat_9', u'feat_10', u'feat_11', u'feat_12', u'feat_13', u'feat_14', u'feat_15', u'feat_16', u'feat_17', u'feat_18', u'feat_19', u'feat_20', u'feat_21', u'feat_22', u'feat_23', u'feat_24', u'feat_25', u'feat_26', u'feat_27', u'feat_28', u'feat_29', u'feat_30', u'feat_31', u'feat_32', u'feat_33', u'feat_34', u'feat_35', u'feat_36', u'feat_37', u'feat_38', u'feat_39', u'feat_40', u'feat_41', u'feat_42', u'feat_43', u'feat_44', u'feat_45', u'feat_46', u'feat_47', u'feat_48', u'feat_49', u'feat_50', u'feat_51', u'feat_52', u'feat_53', u'feat_54', u'feat_55', u'feat_56', u'feat_57', u'feat_58', u'feat_59', u'feat_60', u'feat_61', u'feat_62', u'feat_63', u'feat_64', u'feat_65', u'feat_66', u'feat_67', u'feat_68', u'feat_69', u'feat_70', u'feat_71', u'feat_72', u'feat_73', u'feat_74', u'feat_75', u'feat_76', u'feat_77', u'feat_78', u'feat_79', u'feat_80', u'feat_81', u'feat_82', u'feat_83', u'feat_84', u'feat_85', u'feat_86', u'feat_87', u'feat_88', u'feat_89', u'feat_90', u'feat_91', u'feat_92', u'feat_93']
(X,y,Xtest,labels) = prepareDataset(nsamples='shuffle',standardize=True,log_transform=True,addFeatures=True,doSVD=None,final_filter=None)
best = fmin(fn=func_xgb,space=space_xgb,algo=tpe.suggest,max_evals=50,rseed=123)
#best = fmin(fn=func_nn,space=space_nn,algo=tpe.suggest,max_evals=30,rseed=123)
print best
"""

np.random.seed(12345)
categoricals = [ 'supplier', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x', 'component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5', 'component_id_6', 'component_id_7', 'component_id_8', 'spec1', 'spec2', 'spec3', 'spec4', 'spec5', 'spec6', 'spec7', 'spec8']
categoricals_nospec = [ 'supplier', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x', 'component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5', 'component_id_6', 'component_id_7', 'component_id_8']
categoricals_small = [ 'supplier', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x']
base_cols = ['supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x']
comp_cols = ['component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5', 'component_id_6', 'component_id_7', 'component_id_8']
spec_cols = ['spec1','spec2','spec3','spec4','spec5','spec6','spec7','spec8']
numerical_cols = ['annual_usage','min_order_quantity','quantity','diameter','wall','length','num_bends','bend_radius','num_boss','num_bracket','other','quantity_1','quantity_2','quantity_3','quantity_4','quantity_5','quantity_6','quantity_7','quantity_8','year','month']
log_cols = ['annual_usage','min_order_quantity','quantity','diameter','wall','length','num_bends','bend_radius']
new_feats = ['nspecs', 'nparts', 'max_quantity', 'mean_quantity', 'n_positions']
new_feats2=['cost_per_period', 'quantity_sum_supp', 'annual_usage_sum_supp', 'diameter_sum_supp', 'quantity_size_supp', 'quantity_mean_supp', 'annual_usage_mean_supp', 'diameter_mean_supp', 'quantity_sum_mat', 'annual_usage_sum_mat', 'diameter_sum_mat', 'quantity_size_mat', 'quantity_mean_mat', 'annual_usage_mean_mat', 'diameter_mean_mat']
    

scoring_func = make_scorer(root_mean_squared_error, greater_is_better = False)
counter=0


#XGB
#Xtest,X,y,idx,ta,_ =prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,removeRare=10,oneHotenc=['supplier'],createFeatures=True,createSupplierFeatures=['supplier','quantity'],createVerticalFeatures=True,shapeFeatures=True,timeFeatures=True,materialCost=True,removeLowVariance=True,removeSpec=True)
#Xtest,X,y,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,standardize=None,balance=base_cols+comp_cols+spec_cols,bagofwords=None,createFeatures=True,createVerticalFeatures=True,oneHotenc=['supplier'],removeRare=10,removeSpec=True,dropFeatures=None)
#best = fmin(fn=func_xgb,space=space_xgb,algo=tpe.suggest,max_evals=50,rseed=123)

#NN
Xtest,X,y,idx,ta,_ = prepareDataset(seed=123,nsamples='shuffle',log1p=True,loadBN='load',removeLowVariance = True)
X,Xtest = removeCorrelations(X,Xtest,0.995)
#interact_analysis(X)
#dropFeatures=None#['supplier_2','supplier_45','supplier_40','supplier_33','supplier_23','supplier_21','supplier_8','supplier_7','supplier_3','supplier_1','quantity_8','supplier_11','supplier_0','supplier_27','supplier_28','supplier_34','supplier_6','component_id_8','material_id_1','supplier_44','component_id_7','quantity_7']
#Xtest,X,y,idx,ta,_ = prepareDataset(seed=1234,nsamples='shuffle',log1p=True,balance=base_cols+comp_cols+spec_cols,logtransform=log_cols+new_feats,createFeatures=True,createVerticalFeatures=False,standardize=numerical_cols+new_feats,oneHotenc=categoricals_nospec,removeRare=25,removeSpec=True,removeLowVariance=True)   
print X.describe()
#showCorrelations(X)
scaler = StandardScaler()
X = scaler.fit_transform(X.values)

#interact_analysis(X)
#X = X.values
#y = y.reshape((y.shape[0],1))
best = fmin(fn=func_nn,space=space_nn,algo=tpe.suggest,max_evals=200,rseed=1234)




