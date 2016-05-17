#!/usr/bin/Rscript

###GENERAL SETTINGS###
setwd("D:/COSMOquick/mp_model")
source("D:/data_mining/qspr.R")

set.seed(1234)

fdata<-loadData("D:/COSMOquick/mp_model/ons_physprop_curated.csv") # data combined from ONS and from solubility chap

print(summary(fdata))
idx<-which(fdata$ID!=3) # all except test set
idx_test<-which(fdata$ID==3)
tdata<-fdata[idx_test,]
fdata<-fdata[idx,]
rlist<-prepareData_standard(fdata,removeZeroCols=T,useInterval=F,173,573,removeCols=TRUE)
smiles<-rlist[[2]]
mpdata<-rlist[[1]]
X<-mpdata[,1:ncol(mpdata)-1]
y<-mpdata[,ncol(mpdata)]

y_test<-tdata[,"mpK"]

trainDBN(X,y)
