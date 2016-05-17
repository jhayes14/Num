#!/usr/bin/Rscript
gaFeatureSelection<-function(lX,ly) {
  require(GA)
  popsize=50
  initPop<-fillPop(lX,ly,popsize)
  GARES <- ga("binary", fitness = fitnessFNC,x=lX,y=ly,popSize=popsize, maxiter=20,seed=42, nBits = ncol(lX), pcrossover = 0.8,pmutation = 0.1,monitor = plot,suggestions=initPop)
  plot(GARES)
  print(GARES)
  print(summary(GARES))
  cat("Solution:")
  soln<-as.vector(attr(GARES,"solution"))
  cat(soln,"\n")
  Xs<-lX[,which(soln == 1)]
  for (i in 1:ncol(Xs)) {
    cat("\"",names(Xs)[i],"\"",sep="")
    if (i!=ncol(Xs)) cat(",",sep="")
  }
  return(Xs)
}

fitnessFNC <- function(bitstring,x,y) {
  lossfnc="rmse"
  niter=250
  #cat(bitstring,"\n")
  inc <- which(bitstring == 1)      
  Xs<-x[,inc] 
  #print(summary(Xs))
  #cat("c(")
  #for (i in 1:ncol(Xs)) {
  #  cat("\"",names(Xs)[i],"\"",sep="")
  #  if (i!=ncol(Xs)) cat(",",sep="")
  #}
  #cat(")\n")
  #print(system.time(model<-trainRF(Xs,y,niter,verbose=F)))
  #print(system.time(model<-linRegTrain(Xs,y,NULL,F)))#we need cross-validation here!
  #model<-trainRF(Xs,y,niter,verbose=F)
  if (lossfnc=="auc") {
    tmp<-model$votes[,2]
    tmp<-as.numeric(as.character(tmp))
    loss<-computeAUC(tmp,y,F) 
    cat("AUC:",loss,"\n")
  } else {
    oinfo(Xs)
    oob_preds<-xval_oob(Xs,y,nrfolds=8,repeatcv=5,lossfn="rmse",method="linear",oobfile=NULL)
    loss<-compRMSE(oob_preds,y)
  
    #loss<-compRMSE(model$predicted,model$y)#RF
    #loss<-compRMSE(model$fitted.values,y)#linear
    #print(model)
    #print(summary(model))
    cat("RMSE:",loss,"\n")

    #usually fitness gets MAXIMIZED
    loss<--loss
  }
  #write.table(data.frame(auc=auc,t(bitstring)),file="gabits.csv",sep=",",row.names=FALSE,col.names=F,append=T)
  #return(max(model$cvm))
  return(loss)  
}


fillPop<-function(lX,ly,popsize) {
  ##create initial population  
  initPop <- matrix(0, popsize, ncol(lX))
  initPop<-apply(initPop, c(1,2), function(x) sample(c(0,1),1))
  return(initPop)
}