#!/usr/bin/Rscript
require(nnet)
require(lattice)
require(foreach)
require(doSNOW)

###############################################################################
# Bagged Neural Net                                                           #
# based on code by shea Parkes posted on Kaggle                               #
# seee also http://www.kaggle.com/files/2676/DS06_nnet_Bagstack%20-%20Share.R #
# modified by Chrissly31415                                                   #
###############################################################################

bagged_net<-function(Xtrainl,y,Xtestl=NULL,hiddenl=50,nprocs=8,iterations=10,lossfn="auc",idtrain=NULL,idtest=NULL,verbose=F) {
  if (lossfn=="logloss") {
    loss_fnc<-compLOGLOSS
    loss_str="LOGLOSS"
  } else if (lossfn=="rmse") {
    loss_fnc<-compRMSE
    loss_str="RMSE"
  } else {
    loss_fnc<-computeAUC
    loss_str="AUC"
  }  
  cl<-makeCluster(nprocs, type = "SOCK",outfile="")
  registerDoSNOW(cl)
  bagged.nnet <- foreach(i=1:iterations,.packages='nnet',.verbose=verbose,.export=c("computeAUC","compRMSE","compLOGLOSS")) %dopar% {  
    set.seed(i+42)
    ##Combine train set and test set(for prediction)
    #trainidx<-c(rep(TRUE,nrow(Xtrainl)),rep(FALSE,nrow(Xtestl)))
    #Xmerged<-rbind(Xtrainl,Xtestl)
    bagsize=nrow(Xtrainl)
    bagidx <- sample.int(nrow(Xtrainl),bagsize,replace=TRUE)
    cat("BAGSIZE:",dim(Xtrainl[bagidx,])," ")
    oobidx <- c(!(seq(1,nrow(Xtrainl)) %in% bagidx))
    cat("ITER:",i," OOBSIZE:",dim(Xtrainl[oobidx,]))
    curr.decay<-rlnorm(1,log(2),0.5)
    #curr.decay<-rlnorm(1,-1.0,sdlog=0.5)
    #curr.decay<-5e-4
    #curr.decay<-0.0
    cat(" DECAY:",curr.decay)
    nnet.fit <- nnet(y=y[bagidx],x=Xtrainl[bagidx,],linout=if (lossfn=="rmse") TRUE else FALSE,entropy=if (lossfn=="rmse") FALSE else TRUE,size=hiddenl,decay=curr.decay,maxit=1500,MaxNWts=5000,trace=FALSE)
    #nnet.fit <- nnet(y=y[bagidx],x=Xtrain[bagidx,], size = hiddenl, rang = 0.1, decay = curr.decay, maxit = 200)
    str(nnet.fit)
    pred<-rep(NA,nrow(Xtrainl))
    pred[oobidx]<-predict(nnet.fit,Xtrainl[oobidx,])  
    cat(" LOSS:",loss_fnc(pred[oobidx],y[oobidx]),"\n")
    
    test.pred<-predict(nnet.fit,Xtestl)
    return(list(oob.pred=pred,decay=curr.decay,test.pred=test.pred))
  }
  stopCluster(cl)
  res.decay <- sapply(bagged.nnet,function(x) x$decay)
  #cat("decay",res.decay,"\n")
  res.oob.pred <- sapply(bagged.nnet,function(x) x$oob.pred)
  #print(summary(res.oob.pred))
  #cat("res.oob.pred",res.oob.pred,"\n")
  res.test.pred <- sapply(bagged.nnet,function(x) x$test.pred)
  res.list.oob.pred <- lapply(bagged.nnet,function(x) x$oob.pred)
  #str(res.list.oob.pred)
  ##A single oob prediction for all
  ll<-apply(res.oob.pred,1,mean,na.rm=TRUE)
  cat("###",loss_str,"(NNET) OOB:",loss_fnc(ll,y),"\n")
  res.oob.loss <- apply(res.oob.pred,2,function(x) loss_fnc(x,y))
  ##Logloss for each oob learner
    
  loss_acc<-rep(0.0,times=ncol(res.oob.pred))
  for (idx in 1:ncol(res.oob.pred)) {
    ldf<-res.oob.pred[,1:idx,drop=F]
    #print(summary(ldf))
    pmean<-apply(ldf, 1, function(x) mean(x,na.rm=TRUE))
    #print(summary(pmean))
    auc<-loss_fnc(pmean,y)
    loss_acc[idx]<-auc
  }
  #cat(auc_acc,"\n")
  plot(x=loss_acc,xlab="iterations",ylab="accumulated loss")
  plot(x=res.decay,y=res.oob.loss)	
  
  #write out data
  if (!is.null(idtrain) && !is.null(idtest)) {  
      tmp<-apply(res.test.pred,1,mean,na.rm=TRUE)
      submit<-data.frame(urlid=idtest,label=tmp)
      print(summary(submit))
      write.table(submit, file = "sub3010a.csv", sep = ",", row.names=FALSE,col.names=TRUE)  
      pred<-data.frame(urlid=idtrain,label=ll)
      final<-rbind(submit,pred)  
      write.table(final, file = "nnet.csv", sep = ",", row.names=FALSE,col.names=TRUE)
  }
  
  hist(res.decay,20)
  ##Make a few buckets for the decay values
  curr.decay.buckets <- cut(res.decay,breaks=quantile(res.decay,seq(0,1,length.out=20)),include.lowest=TRUE)
  #cat("db:",curr.decay.buckets,"\n")
  ##Show the buckets from an ensemble perspective
  opt.decay.bucket <- tapply(res.list.oob.pred,list(curr.decay.buckets),function(x) loss_fnc(apply(do.call(cbind,x),1,mean,na.rm=TRUE),y) )
  dotchart(opt.decay.bucket,main='Bagged NNet Performance by Decay Bucket',xlab='OOB Loss')
  
  ##Show the buckets from an individual basis
  opt.decay.bucket.indiv <- tapply(
    ##List of oobs
    res.list.oob.pred
    ##Buckets of decay values
    ,list(curr.decay.buckets)
    ,function(x) mean(sapply(x,function(x1) loss_fnc(x1,y)),na.rm=TRUE,trim=0.1)
  )
  #opt.decay.bucket.indiv
  dotchart(opt.decay.bucket.indiv,main='Individual NNet Performance by Decay Bucket',xlab='OOB Loss')
  
  ###Overlay the ensemble and individual
  overlay.frame <- data.frame(
    decay.range=factor(levels(curr.decay.buckets),levels=levels(curr.decay.buckets))
    ,ensemble=opt.decay.bucket
    ,indiv=opt.decay.bucket.indiv
  )
  #print(summary(overlay.frame))
  dotplot(decay.range~ensemble,data=overlay.frame,auto.key=TRUE,type='b',xlab='OOB Log-Loss',main='Tuning NNet Decay for Stacking') 
}
