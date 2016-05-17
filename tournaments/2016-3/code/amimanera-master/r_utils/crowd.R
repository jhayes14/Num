#!/usr/bin/Rscript

###GENERAL SETTINGS###
set.seed(1234)

library(randomForest)

if (.Platform$OS.type=='unix') {
  setwd(".")
  source("./qspr.R")
} else {
  setwd("D:/COSMOquick/mp_model")
  source("D:/data_mining/qspr.R")
}


Xtrain <- read.csv(file="../data/Xtrain.csv")
ytrain <- read.csv(file="../data/ytrain.csv")

ytrain<-as.factor(ytrain$class)
oinfo(ytrain)

print(summary(Xtrain))
print(summary(ytrain))
oinfo(ytrain)

#Xtrain<-boruta_select(Xtrain,ytrain)

rf1 <- randomForest(Xtrain,ytrain,ntree=500,importance = T)
imp<-importance(rf1, type=1)
print(imp)
varImpPlot(rf1,n.var=50,type=1,main="")
print(rf1)

