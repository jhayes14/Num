#!/usr/bin/Rscript
rf_select<-function(lX,ly,iter=250,nvar=5) {
  require(randomForest)
  cat("Selection of variables by RF importance (increase in MSE due to permutation of variables):\n")
  mydata=data.frame(lX,target=ly)
  print(summary(mydata))
  mydata.rf <- randomForest(lX,ly,ntree=iter,importance = T,nodesize=10,do.trace=25)
  
  if (mydata.rf$type=="regression") {
    cat("RF regression\n")
    nr.samples<-nrow(lX)
    rmse<-mean(sqrt(mydata.rf$mse)*(nr.samples-1)/nr.samples)
    stdev<-sd(mydata.rf$mse)
    cat("RF RMSE:",rmse,"stddev:",stdev," ")
    cat("R^2:",mean(mydata.rf$rsq),"\n")
  }
  
  varImpPlot(mydata.rf,main="Random Forest Variable Importance")
  increaseMSE<-data.frame(importance(mydata.rf, type=1))  
  index<-order(-increaseMSE[,1])
  dd<-increaseMSE[index,,drop=F]
  #filter variable which do no contribute 
  print(dd)
  str(dd)
  if (mydata.rf$type=="regression") {
    dd<-dd[dd$X.IncMSE>0.0,,drop=F]
  }  else {
    dd<-dd[dd$MeanDecreaseAccuracy>0.0,,drop=F]
  }
  
  #select top nvar
  dd<-dd[1:nvar,,drop=F]
  validCols<-rownames(dd)  
  lX<-subset(lX,select=validCols)
  
  cat("\n####Selection: \n")
  for (i in 1:ncol(lX)) {
    cat("\"",names(lX)[i],"\"",sep="")
    if (i!=ncol(lX)) {
      cat(",",sep="")
    } else {
      cat("\n\n")
    }
  }
  
  #print(summary(lX))
  return(lX) 
  #str(mydata.rf$importance)
  #print(summary(mydata.rf$importance))
  #print(rownames(mydata.rf$importance))
  #print(mydata.rf$importance[,1])
  #return(tmpframe)
}