#!/usr/bin/Rscript

###GENERAL SETTINGS###
set.seed(1234)

if (.Platform$OS.type=='unix') {
  setwd(".")
  source("./qspr.R")
} else {
  setwd("D:/COSMOquick/mp_model")
  source("D:/data_mining/qspr.R")
}

###READ & MANIPULATE DATA###

#fdata<-loadData("D:/COSMOquick/mp_model/ons_curated.csv")
#<-loadData("D:/COSMOquick/mp_model/litdata_small_mod.csv") # data from solubility chap
# #fdata<-loadData("D:/COSMOquick/mp_model/ons_physprop_curated.csv") # data combined from ONS and from solubility chap
# fdata<-loadData("D:/COSMOquick/mp_model/ons_physprop_curated_win.csv") # data combined from ONS and from solubility chap, descriptors from linux
# #fdata<-loadData("D:/COSMOquick/mp_model/litdata_zwi.csv") # turned amino acids into zwitter ions
# #fdata<-loadData("D:/COSMOquick/mp_model/litdata_fcos.csv")
# #fdata<-loadData("D:/COSMOquick/mp_model/litdata_cdk.csv")
# print(summary(fdata))
# idx<-which(fdata$ID==2 | fdata$ID==0)# physprop
# #idx<-which(fdata$ID!=3) # all except test set
# #idx<-which(fdata$ID>3) # remaining ONS only data
# idx_test<-which(fdata$ID==3)
# tdata<-fdata[idx_test,]
# fdata<-fdata[idx,]
# #fdata<-loadData("D:/COSMOquick/mp_model/ons_moments.csv")
# #fdata<-loadData("D:/COSMOquick/mp_model/acree_standard_small.csv")
# rlist<-prepareData_standard(fdata,removeZeroCols=T,useInterval=F,173,573,removeCols=TRUE)
# #rlist<-prepareData_acree(fdata,173,573,F)
# smiles<-rlist[[2]]
# mpdata<-rlist[[1]]
# X<-mpdata[,1:ncol(mpdata)-1]
# y<-mpdata[,ncol(mpdata)]
# #d<-scan(n=1)
# print(summary(X))
# #X<-removeColVar(X,0.95)
# #X<-normalizeData(X)


###ANALYSIS TOOLS###
#cat("correlation:",cor(X$tmult,y),"\n")
#compareDataSets()
#pc_analysis(Xlin,y)
#compareViaPrinComp(Xlin,Xlin,y,273)
#genLinMod(X_red,y)
#Xrf<-outlierDetection(Xlin,y,2,sortOnResidues=TRUE,TRUE,fit,smiles)
#linRegPredict(fit,X_test,y_test,smiles_test)
#mlr_analysis(fit,X,smiles,"ClC(Cl)(Cl)Br")
#mlr_analysis(fit,X,smiles,"c1cnc(cn1)C(=S)N")
#mlr_analysis(fit,X,smiles,"CCCCCCCC")
#mlr_analysis(fit,X,smiles,"CCCCCCCCCCCCCCCCCCCC")
#gam_model(X,y,T)
#xvalSVM(X,y)
#trainSVM(X,y)

###RESIDUE MODEL###
#rf<-trainResidues(Xlin,y,fit,ntree)
#resmodel<-trainResidues(X,y,ntree)
#predictResidues(Xrf,y,rf,fit)



###FEATURE SELECTION###
#Xlin<-variableSelection(X,y,"backward",40)

#Xrf<-boruta_select(X,y)
#rf,gam,svm,boost
#Xrf<-greedySelect(Xlin,y,20,"randomForest","rmse")
#Xrf<-rf_select(Xlin,y,300,14)
#gaFeatureSelection(Xlin,y)
#selectedCols<-c("h_hb","area","M2","M6","Macc1","Mdon1","Mdon4","ringbonds","alkylgroups","internal_hbonds","conjugated_bonds","rotbsdmod","tmult","nbr11","N_total","zwitterion_in_water")#hfus ga selection
#selectedCols<-c("M2", "ringbonds", "rotbsdmod", "h_int", "Mdon3","N_total", "conjugated_bonds", "tmult", "zwitterion_in_water", "alkylgroups", "rotatable_bonds", "molweight", "internal_hbonds", "rbwring", "Mdon1") #Original descriptors CQ
#selectedCols<-c("Mdon3","M2", "ringbonds", "rotbsdmod", "h_int", "Mdon2", "conjugated_bonds", "tmult", "zwitterion_in_water", "alkylgroups", "rotatable_bonds","area", "rbwring")#final selection litdata model without N_total
#selectedCols<-c("Mdon3","M2", "ringbonds", "rotbsdmod", "h_int", "Mdon2","N_total", "conjugated_bonds", "tmult", "zwitterion_in_water", "alkylgroups", "rotatable_bonds","area", "rbwring")#final selection litdata model 
#selectedCols<-c("molweight","M2","rotatable_bonds","conjugated_bonds","Mdon3","ringbonds","N_total","h_int","zwitterion_in_water","tmult","rotbsdmod","alkylgroups","internal_hbonds","nbr11","rbwring","volume","alkylatoms","avratio","Mdon1","Macc2")# iterative rmse=41.2
#selectedCols<-c("molweight","M2","rotatable_bonds","conjugated_bonds","M6","E_Ring","N_total","Mdon3","tmult","massprotbond","avratio","zwitterion_in_water","rotbsdmod","rbwring","h_int","alkylatoms","Macc1","alkylgroups","internal_hbonds","Mdon1")#new sets greedy
#recent:c("molweight","M2","massprotbond","conjugated_bonds","h_hb","E_Ring","Mdon1","N_total","tmult","h_vdw","zwitterion_in_water","rbwring","avratio","rotbsdmod","internal_hbonds","N_amino","ringbonds","alkylatoms","Macc1","nbr11")

#Xrf<-subset(Xlin,select=selectedCols)
#cat(">>Number of features:",ncol(Xrf),"\n")
#cat(">>Number of samples:",nrow(Xrf),"\n")

###LINEAR MODEL###
#print(summary(Xlin))
#fit<-linRegTrain(Xlin,y,smiles,T)
#fit<-linRegTrain(Xrf,y,smiles,T)
#xvalid(Xrf,y,nrfolds=10,modname="linear",lossfn="rmse",ntree)
#saveMLR(fit,"Tm[k]")

###NEURAL NET###
#hiddenl=10
#nprocs=4
#iterations=ntree
#Xtest<-matchByColumns(Xrf,tdata)
#bagged_net(Xrf,Xtest,y,hiddenl,nprocs,iterations,lossfn="rmse",orthog=FALSE,idx,idx_test)


#how can we describe adamantan and co? ring density?
#ORIG RMSE=41.84
#masspbond<-Xrf$rotatable_bonds/Xlin$molweight#RMSE=41.69
#ringdensity<-Xlin$ringbonds/Xlin$volume#42.0368
#ringdensity<-Xlin$tmult/(Xlin$rbwring+1)#41.82
#Xrf<-subset(Xrf,select=-rbwring)
#masspbond<-Xrf$rotatable_bonds/Xlin$area#41.83
#masspbond<-Xlin$rotatable_bonds/Xlin$volume#RMSE=41.91
#masspbond<-Xlin$rotbsdmod/Xlin$molweight#RMSE=41.74

#ringdensity<-Xlin$molweight/(Xlin$rbwring+1)#41.74
#ringdensity<-Xlin$rbwring/(Xlin$molweight)#41.74
#ringdensity<-Xlin$molweight/(pmax(Xlin$rbwring,1))#41.81
#Xrf<-cbind(ringdensity,Xrf)
#massprotbond<-Xlin$molweight/(Xlin$rotatable_bonds+1)#RMSE=41.60
#massprotbond<-Xlin$molweight/(Xlin$rotatable_bonds+1)
#massprotbond<-Xlin$molweight/(pmax(Xlin$rotatable_bonds,1))#RMSE=41.66
#Xrf<-cbind(massprotbond,Xrf)
#Xrf<-subset(Xrf,select=-molweight)
#cat(">>Number of features:",ncol(Xrf),"\n")
#cat(">>Number of samples:",nrow(Xrf),"\n")

###TRAIN GBM###
# interaction.depth = 4, n.trees = 10000 and shrinkage = 0.05.   RMSE=39.27
# interaction.depth = 5, n.trees = 10000 and shrinkage = 0.01.  RMSE=39.21
#
#ntree<-1000
#intdepth = 4
#sh = 0.05
# RMSE=0.3863
#gbm_grid(Xrf,y,lossfn="rmse") 
#rf<-gbm(y~.,data.frame(Xrf,y=y),distribution="gaussian",n.trees=ntree,interaction.depth=intdepth,shrinkage=sh,verbose=F,cv.folds=5)
#rf<-gbm.fit(Xrf,y,distribution="gaussian",n.trees=ntree,interaction.depth=intdepth,shrinkage=sh,verbose=T)
#xvalid(Xrf,y,nrfolds=5,"gbm",loss="rmse",iter=ntree)
#ntree_opt <- gbm.perf(rf, met#fdata<-loadData("D:/COSMOquick/mp_model/ons_physprop_curated.csv") # data combined from ONS and from solubility chap
# fdata<-loadData("D:/COSMOquick/mp_model/ons_physprop_curated_win.csv") # data combined from ONS and from solubility chap, descriptors from linux
# #fdata<-loadData("D:/COSMOquick/mp_model/litdata_zwi.csv") # turned amino acids into zwitter ions
# #fdata<-loadData("D:/COSMOquick/mp_model/litdata_fcos.csv")
# #fdata<-loadData("D:/COSMOquick/mp_model/litdata_cdk.csv")
# print(summary(fdata))
# idx<-which(fdata$ID==2 | fdata$ID==0)# physprop
# #idx<-which(fdata$ID!=3) # all except test set
# #idx<-which(fdata$ID>3) # remaining ONS only data
# idx_test<-which(fdata$ID==3)
# tdata<-fdata[idx_test,]
# fdata<-fdata[idx,]
# #fdata<-loadData("D:/COSMOquick/mp_model/ons_moments.csv")
# #fdata<-loadData("D:/COSMOquick/mp_model/acree_standard_small.csv")
# rlist<-prepareData_standard(fdata,removeZeroCols=T,useInterval=F,173,573,removeCols=TRUE)
# #rlist<-prepareData_acree(fdata,173,573,F)
# smiles<-rlist[[2]]
# mpdata<-rlist[[1]]
# X<-mpdata[,1:ncol(mpdata)-1]
# y<-mpdata[,ncol(mpdata)]
# #d<-scan(n=1)
# print(summary(X))
#X<-removeColVar(X,0.95)
#X<-normalizeData(X)hod = "cv", plot.it = T)
#cat("Best ntree:",ntree_opt,"/",ntree)
#saveGBM(rf,"Tm[kcal/mol")

###CREATE OOB RESULTS
#oobres<-gbm_xval(Xrf,y,numberTrees=15000,nrfolds=4,intdepth=6,sh=0.01,repeatcv=10)
#oobres<-data.frame(urlid=smiles,pred=oobres,y=y)
#write.table(oobres,file="oob2.csv",sep=",",row.names=FALSE)

###TRAIN RF###
#ntree<-300
#mtry=5
#rf<-trainRF(Xrf,y,iter=ntree,m.try=mtry,node.size=5, verbose=T,fimportance=T)
#printErrors(rf,Xrf,y,smiles)
#write.table(predict(rf,X),file="prediction_inbag.csv",sep=",",row.names=FALSE)
#xvalid(Xrf,y,5,modname="rf",lossfn="rmse",ntree)
#print(varUsed(rf, by.tree=FALSE, count=TRUE))
#saveRF(rf,"Tm[K]")

#correlationEllipses(cor(data.frame(Xrf,Tm=y)))
#makeBubblePlot(rf,Xrf,y)
#cat(toBibtex(citation()),"\n")
#cat(toBibtex(citation(package="gbm")),"\n")
#cat(toBibtex(citation(package="randomForest")),"\n")
warnings()
