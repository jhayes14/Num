#!/usr/bin/Rscript
###GENERAL SETTINGS###
set.seed(41)

if (.Platform$OS.type=='unix') {
  setwd(".")
  source("./qspr.R")
} else {
  setwd("D:/COSMOquick/mp_model")
  source("D:/data_mining/qspr.R")
}


borutaRun<-function(Xtrain,ytrain) {
  print(summary(ytrain))
  Xrf<-boruta_select(Xtrain,ytrain$P)
}

submitFunction<-function(models,Xtest,submission) {
    predictions <- sapply(models,predict,newdata=Xtest)
    colnames(predictions) <- c("Ca","P","pH","SOC","Sand")
    submission <- cbind(PIDN=submission,predictions)
    write.csv(submission,"/home/loschen/Desktop/datamining-kaggle/african_soil/submissions/sub0510a.csv",row.names=FALSE,quote=FALSE)
}


mainRoutine<-function() {
    ###READ & MANIPULATE DATA###
    Xtrain<-read.csv("/home/loschen/Desktop/datamining-kaggle/african_soil/training.csv",sep=",",header=TRUE,stringsAsFactors=FALSE)
    Xtest<-read.csv("/home/loschen/Desktop/datamining-kaggle/african_soil/sorted_test.csv",sep=",",header=TRUE,stringsAsFactors=FALSE)
    #Xtrain<-loadData("/home/loschen/Desktop/datamining-kaggle/african_soil/training_mod.csv",separator=",")
    #Xtest<-loadData("/home/loschen/Desktop/datamining-kaggle/african_soil/test_mod.csv",separator=",")
    submission <- Xtest[,1]
    
    ytrain <- Xtrain[,c("Ca","P","pH","SOC","Sand")]
    
    Xtrain <- Xtrain[,2:3595]#all
    Xtest <- Xtest[,2:3595]
    #Xtrain <- Xtrain[,2:3579]#only spectra
    #Xtest <- Xtest[,2:3579]
    
    #idx<-sample(nrow(Xtrain), 10y0)
    #Xtrain<-Xtrain[idx,]
    #ytrain<-ytrain[idx,]

    #Xtrain<-Xtrain[,c("BSAN","BSAS","BSAV","CTI","ELEV","EVI","LSTD","LSTN","REF1","REF2","REF3","REF7","RELI","TMAP","TMFI")]
    Xall<-rbind(Xtest,Xtrain)
    Xall$Depth <- ifelse(Xall$Depth == 'Topsoil',1,0)
    
    
    Xall<-removeColVar(Xall,0.9999)
    #Xall<-normalizeData(Xall)#matrix????????
    Xtest<-Xall[1:nrow(Xtest),]
    oinfo(Xtest)
    Xtrain<-Xall[(nrow(Xtest)+1):nrow(Xall),]
    oinfo(Xtrain)
    
    #gaFeatureSelection(Xtrain,ytrain[,1])
    #quit()
    
    results<-lapply(1:ncol(ytrain),
		  function(i)
		  {
		    #svm(train,labels[,i],cost=10000,scale=FALSE)
		    prop<-colnames(ytrain)[i]
		    filename<-paste(prop,"_oob.csv",sep="")
		    cat("\n##TARGET:",prop,"\n")
		    
		    #fit<-linRegTrain(Xlin,ytrain[,i],NULL,T)
		    #score<-xvalid(Xlin,ytrain[,i],nrfolds=8,modname="linear",lossfn="rmse")
		    #oob_preds<-xval_oob(Xtrain,ytrain[,i],nrfolds=8,repeatcv=10,lossfn="rmse",method="linear",iterations=50,oobfile=NULL)
		    #score<-compRMSE(oob_preds,ytrain[,i])
		    
		    #hist(oob_preds)
		    #modelbuilding
		    
		    #GA
		    #Xlin<-gaFeatureSelection(Xtrain,ytrain[,i])
		    #score<-xvalid(Xlin,ytrain[,i],nrfolds=8,modname="linear",lossfn="rmse")
		    
		    
		    #gen lasso
		    D = diag(1,p)
		    out = genlasso(y, X=X, D=D)
		    #Xlin<-variableSelection(Xtrain,ytrain[,i],"forward",50,plotting=TRUE)
		    #model<-linRegTrain(Xlin,ytrain[,i],NULL,F)
		    
		    #prediction
		    #Xtest<-matchByColumns(Xlin,Xtest)
		    #preds<-predict(model,Xtest)
		    #preds<-data.frame(prediction=preds)
		    #preds<-rbind(preds,oob_preds)
		    #write.table(preds,file=filename,sep=",",row.names=FALSE)
		    
		    
		    #print(#score)
		    #return(model)
		    return(score)
		  })
    cat("Final RMSE:",mean(unlist(results))," stdev: ",sd(unlist(results)),"\n")
    results[[2]]<-NULL#setting P to zero
    cat("Final RMSE(NO P):",mean(unlist(results))," stdev: ",sd(unlist(results)),"\n")
    #submitFunction(results,Xtest,submission)
}               


mainRoutine()
