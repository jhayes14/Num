greedySelect<-function(Xs,ly,itermax=10,method="randomForest",losstype="rmse",good_features=NULL,repeatcv=2,nrfolds=4,treeiter=250) {
  require(foreach)
  require(doSNOW) 
  cat("Initial:",good_features,"\n")
  good_features<-match(good_features,colnames(Xs))
  loss_acc<-mat.or.vec(ncol(Xs),1)
  for (i in 1:itermax) {
    #if (i>(itermax-length(good_features))) next
    cat("\n####### Searching step: ",i,"\n")
    auclist<-mat.or.vec(ncol(Xs),1)
    for (j in 1:ncol(Xs)) {
      if (j %in% good_features) {
        auclist[j]<-0.0
        if (losstype=="rmse") auclist[j]<-1e15
        next
      } 
      act_features<-c(good_features,j)
      cat("Using variables:",act_features," ")
      Xa<-Xs[,act_features,drop=F] # we do not want a vector
      cat(colnames(Xa),"\n")   
      if (method=='randomForest') {
        cl<-makeCluster(repeatcv, type = "SOCK",outfile="")
        registerDoSNOW(cl)
        lossdat<-foreach(k = 1:repeatcv,.packages='randomForest',.combine="rbind",.export=c("computeAUC","trainRF")) %dopar% {
          model<-trainRF(Xa,ly,treeiter)
          if (losstype=="auc") {
            tmp<-model$votes[,2]
            tmp<-as.numeric(as.character(tmp))
            loss<-computeAUC(tmp,model$y,F)
            #auc_all<-apply(cvglm$fit.preval, 2, function(x) computeAUC(x,ly))
            #cat("AUC best:",max(auc_all)," index: ",which.max(auc_all),"\n")
            #loss<-max(auc_all)
          } else  if (losstype=="rmse") {
            nr.samples<-nrow(Xa)
            loss<-mean(sqrt(model$mse)*(nr.samples-1)/nr.samples)
          }
          return(loss)    
        }
        stopCluster(cl)
        loss<-mean(lossdat[,1])
      }
      #only parallel in folds
      else {
        tmp <- as.data.frame(Xa)   
        tmp$target <- ly
        model<-glm(target~., data=tmp,family=binomial(link="logit"))
        tmp<-xval_oob(Xa,ly,Xtest=NULL,repeatcv=repeatcv,nrfolds=nrfolds,method='linear',lossfn="auc")
        loss<-computeAUC(tmp$prediction,ly,F)
      }
      cat("Loss (",losstype,") of potential feature: ",j," :",loss," ")
      if (i>1) cat(" Delta (last-actual):",loss_acc[i-1]-loss)
      cat("\n")
      auclist[j]=loss
    }
    if (losstype=="auc") {
      best<-which.max(auclist)
    } else  if (losstype=="rmse") {
      best<-which.min(auclist)
    }
    
    cat("######Best feature:",best," with loss (",losstype,") :",auclist[best]," ")
    good_features<-c(good_features,best)
    Xnew<-Xs[,best,drop=F]
    cat(colnames(Xnew),"\n")
    loss_acc[i]<-auclist[best]
  }
  df<-Xs[,good_features]
  cat("Selected features:",good_features,"\n")
  cat("c(")
  for (i in 1:ncol(df)) {
    cat("\"",names(df)[i],"\"",sep="")
    if (i!=ncol(df)) cat(",",sep="")
  }
  cat(")\n")
  cat("Loss improvement:",loss_acc,"\n")
  plot(loss_acc) 
  write.table(df,file="greedy.csv",sep=",",row.names=FALSE,col.names=T)
  return(df)
}