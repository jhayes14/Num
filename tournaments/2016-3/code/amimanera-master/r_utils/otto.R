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





rf_model<-function(Xtrain,y,Xtest) {
    rf<-trainRF(Xtrain,y,iter=50,m.try=5,node.size=5, verbose=T,fimportance=T)
    pred <- predict(rf,Xtest,type="prob")
    submit <- data.frame(id = Xtest$id, pred)
    write.csv(submit, file = "/home/loschen/Desktop/datamining-kaggle/otto/submissions/submission09042015.csv", row.names = FALSE)
}


###READ & MANIPULATE DATA###

# load data
data <- read.csv("/home/loschen/Desktop/datamining-kaggle/otto/train.csv")
data <-data[sample(nrow(data), 300), ]

Xtest <- read.csv("/home/loschen/Desktop/datamining-kaggle/otto/test.csv")
sample_sub <- read.csv("/home/loschen/Desktop/datamining-kaggle/otto/sampleSubmission.csv")
# remove id column so it doesn't get picked up by the random forest classifier
y<-data[,ncol(data)]
Xtrain<-data[,3:ncol(data)-1]

#print(summary(Xtrain))
#print(summary(y))
#print(summary(Xtest))

#Xrf<-boruta_select(Xtrain,y)#bringt nichts

rf_model(Xtrain,y,Xtest)

warnings()
