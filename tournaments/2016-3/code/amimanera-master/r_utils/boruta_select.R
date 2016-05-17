
boruta_select<-function(lX,ly) {
  library(Boruta)
  mydata=data.frame(lX,target=ly)
  summary(mydata)
  Boruta(target~.,data=mydata,doTrace=2,maxRuns=150)->Boruta.results
  #Nonsense attributes should be rejected
  print(Boruta.results);
  print(summary(Boruta.results))
  cat("Summary:\n")
  print(summary(Boruta.results$finalDecision))

  idx_confirmed<-Boruta.results$finalDecision == "Confirmed"
  idx_tentative<-Boruta.results$finalDecision == "Tentative"
  
  #plotImpHistory(Boruta.results)
  
  #Boruta.tentative<-TentativeRoughFix(Boruta.results, averageOver = "finalRound")
  #validCols<-getSelectedAttributes(Boruta.tentative)
  validCols<-idx_confirmed | idx_tentative
  
  tmpframe<-subset(mydata,select=validCols)
  print(summary(tmpframe))
  #tmpframe<-data.frame(tmpframe,target=mydata$target)
  
  cat("colnames:\n")
  
  for (i in 1:ncol(tmpframe)) {
    cat("\"",names(tmpframe)[i],"\"",sep="")
    if (i!=ncol(tmpframe)) {
	cat(",",sep="")
      } else {
	cat("\n\n")
      }
  }
  
  
  write.table(tmpframe,file="boruta_features.csv",sep=",",row.names=FALSE)
  return(tmpframe)
}

