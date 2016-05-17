################################################################################
#    code collection of helper tools for data mining and machine learning      #
#    (c) by chrissly31415  2014                                                #
################################################################################

require(foreach)
require(doSNOW)

if (.Platform$OS.type=='unix') {
    cat("We are on linux.\n")
    rootdir<-"/home/loschen/calc/amimanera/r_utils"
    source(paste(rootdir,"/boruta_select.R",sep=""))
    source(paste(rootdir,"/rf_select.R",sep=""))
    source(paste(rootdir,"/xvalidation.R",sep=""))
    source(paste(rootdir,"/gaFeatureSelection.R",sep=""))
} else {
    cat("We are on windows.\n")
    
    #rm(list = ls(all = TRUE))
    #in case R was aborted
    if (sink.number()>1) sink()
    args <- commandArgs(TRUE)
    options(error=recover)
    
    rootdir<-"D:/data_mining/amimanera/r_utils"
    source(paste(rootdir,"/xvalidation.R",sep=""))
    source(paste(rootdir,"/boruta_select.R",sep=""))
    source(paste(rootdir,"/rf_select.R",sep=""))
    source(paste(rootdir,"/greedySelect.R",sep=""))
    source(paste(rootdir,"/gaFeatureSelection.R",sep=""))
    source(paste(rootdir,"/bagged_net.R",sep=""))
}



loadData<-function(filename,separator=";") {
  ldata = read.csv(file=filename,sep=separator)
  #ldata = read.csv(file="acree_standard_small.csv",sep=";")  
  #ldata = read.csv(file="ONSMP13_orig2.csv")
  #ldata = read.csv(file="ons_standard_small.csv",sep=";")
  #ldata = read.csv(file="katritzky_n_small.csv",sep=";")
  #ldata = read.csv(file="ons_final.csv")
  #take only non fragmented
  
}

prepareData_standard<-function(ldata,removeZeroCols=T,drugLike=F,outlierRemoval=F,useInterval=F,lowT=0,highT=1000,removeCols=TRUE) {
  #REMOVE ROWS
  cat(">>Number of samples:",nrow(ldata),"\n")
  if (useInterval) {
    cat("Applying T-intervall Filter\n")   
    ldata<-ldata[ldata$mpK<highT,]
    ldata<-ldata[ldata$mpK>lowT,]
    cat(">>Number of samples:",nrow(ldata),"\n")   
  }
  
  #Pseudo lipinski filter
  if (drugLike) {    
    cat("Applying drug-likeliness Filter\n")   
    ldata<-ldata[ldata$molweight>=150,]
    cat(">>Filter MW>=150:",nrow(ldata),"\n")
    ldata<-ldata[ldata$molweight<=500,]
    cat(">>Filter MW<=500:",nrow(ldata),"\n")
    ldata<-ldata[ldata$rotatable_bonds<=7,]
    cat(">>Filter rotbonds<=7:",nrow(ldata),"\n")
    if("Donor" %in% colnames(ldata)) ldata<-ldata[ldata$Donor<=5,]
    if("Acceptor" %in% colnames(ldata)) ldata<-ldata[ldata$Acceptor<=10,] 
    cat(">>Number of samples:",nrow(ldata),"\n")
  }
  
  #remove outliers
  if (outlierRemoval) {
    cat("Outlier removal\n")
    idx<-grep("Si",ldata$SMILES, perl=TRUE)
    #cat(ldata$SMILES[idx])
    ldata<-ldata[-idx,]
    #ldata<-subset(ldata,select=-idx)
    #remove adamantanes
    idx<-grep("adamant",ldata$name, perl=TRUE)
    #cat(ldata$name[idx])
    ldata<-ldata[-idx,]
    cat(">>Number of samples:",nrow(ldata),"\n")
  }

  smiles<-ldata[,2:3]
  ldata<-ldata[,6:length(ldata)]
  #REMOVE COLS
  if (removeCols) {
    ldata<-subset(ldata,select=-fragments)
    ldata<-subset(ldata,select=-Macc4)
    ldata<-subset(ldata,select=-M4)
    #ldata<-subset(ldata,select=-M5)
    #ldata<-subset(ldata,select=-M6)
    ldata<-subset(ldata,select=-Macc3)
    ldata<-subset(ldata,select=-M3)
    ldata<-subset(ldata,select=-frag_quality)
  }
  
  #ldata<-subset(ldata,select=-zwitterion_in_water)
  #ldata<-subset(ldata,select=-similarity)
  #ldata<-subset(ldata,select=-res)
  #ldata<-subset(ldata,select=-alkane) 
  #ldata<-subset(ldata,select=-nbr3s3s)
  #ldata<-subset(ldata,select=-nbr3u1)
  #ldata<-subset(ldata,select=-nbr2s1)
  #ldata<-subset(ldata,select=-nbr3s2s)
  #ldata<-subset(ldata,select=-nbr3s1)
  #ldata<-subset(ldata,select=-nbr3u2u)
  #ldata<-subset(ldata,select=-nbr2u1)
  #ldata<-subset(ldata,select=-mpK)
  #ldata<-subset(ldata,select=-Hfus_exp.kcal.mol.)
  #ldata<-subset(ldata,select=-Sfus_exp.cal.molK.)  
  if (removeZeroCols==T) { 
    cs<-colSums(abs(ldata)==0)
    #cat(cs)
    if (0 %in% cs) {
      ldata<-ldata[,which(colSums((ldata))!=0)]
    }    
  } 
  #cat("Data after preparation:\n")
  return(list(ldata,smiles))
  return(list(data=ldata,smiles=smiles))
}

prepareData_ons<-function(ldata,lowT,highT,testSet) {
  #REMOVE ROWS
  #ldata<-ldata[ldata$Fragments<2,]
  #ldata<-ldata[ldata$Alkane>0,]
  ldata<-ldata[ldata$mpK<highT,]
  ldata<-ldata[ldata$mpK>lowT,]
  #ldata<-ldata[1:100,]
  #ldata<-ldata[ldata$Fragments<2,]
  if (testSet==T) {
    ldata<-ldata[ldata$Alkane>0,]
    #ldata<-ldata[ldata$zwitterion>0,]
    #ldata<-ldata[ldata$Fragments<2,]
  } 
  #remove outliers
  ldata<-ldata[ldata$SMILES!="CCCCC(CCCC)=O",]
  #print(summary(ldata))
  #hist(ldata$mpK,50)
  #ldata<-ldata[ldata$Si.Sn<1,]
  #define matrix
  smiles<-ldata[,2:3]
  ldata<-ldata[,6:length(ldata)]
  
  #ADD COLS
  #entropy100<-ldata$mu_self-ldata$hint_self100
  #ldata<-cbind(entropy100,ldata)
  Txentropy<-(ldata$mu_self-ldata$hint_self)*-1
  ldata<-cbind(Txentropy,ldata)
  
  #hist(ldata$mu_self_noVDW,50)
  #Txentropy_noVDW<-(ldata$mu_self_noVDW-ldata$hint_self_noVDW)*-1
  #ldata<-cbind(Txentropy_noVDW,ldata)
  
  ldata<-subset(ldata,select=-Alkane) 
  #hist(ldata$hb_self100,50) 
  hb_self_lowT<-(pmax(-5,ldata$hb_self100))
  ldata<-cbind(hb_self_lowT,ldata) 
  ldata<-subset(ldata,select=-hb_self)
  h_crs<-(ldata$hint_self-ldata$hint_self100)
  ldata<-cbind(h_crs,ldata) 
  
  #ldata<-subset(ldata,select=-hb_self100)
  #ldata<-subset(ldata,select=-hb_self_noVDW)
  #ldata<-subset(ldata,select=-Rotatable.bonds) 
  #ldata<-subset(ldata,select=-zwitterion) 
  avratio<-ldata$Area/ldata$Volume
  #avratio<-ldata$Area/(ldata$Volume^(2/3))
  ldata<-cbind(avratio,ldata) 
  #alkylatom_groups<-ldata$Alkylatoms/max(1,ldata$Alkylgroups)
  #ldata<-cbind(alkylatom_groups,ldata) 
  #termbonds<-ldata$Rotatable_bond-ldata$X.rotbonds.CDK.
  #ldata<-cbind(termbonds,ldata)
  #expbonds<-(ldata$Rotatable_bond+ldata$X.rotbonds.CDK.)/2.0
  #ldata<-cbind(expbonds,ldata)
  
  #REMOVE COLS
  ldata<-subset(ldata,select=-zwitterion)
  ldata<-subset(ldata,select=-hint_self)
  #ldata<-subset(ldata,select=-Kier2)
  #ldata<-subset(ldata,select=-hint_self_noVDW)
  #ldata<-subset(ldata,select=-mu_self_noVDW)
  #ldata<-subset(ldata,select=-hint_self100)
  #ldata<-subset(ldata,select=-mu_self100) 
  ldata<-subset(ldata,select=-X..moment6)
  ldata<-subset(ldata,select=-X..moment5)
  ldata<-subset(ldata,select=-X..moment4)
  ldata<-subset(ldata,select=-X..moment3)
  ldata<-subset(ldata,select=-X..moment2)
  ldata<-subset(ldata,select=-Fragments)
  #ldata<-subset(ldata,select=-Molweight..g.mol.)
  #ldata<-subset(ldata,select=-MWeight)
  #ldata<-subset(ldata,select=-Alkylatoms)
  #ldata<-subset(ldata,select=-Alkylgroups)
  #ldata<-subset(ldata,select=-Sumq.e.)
  ldata<-subset(ldata,select=-HB.acc_m4)
  ldata<-subset(ldata,select=-HB.acc_m3)
  ldata<-subset(ldata,select=-HB.acc_m1)
  ldata<-subset(ldata,select=-HB.don_m4)
  ldata<-subset(ldata,select=-HB.don_m3)
  ldata<-subset(ldata,select=-HB.don_m1)
  #ldata<-subset(ldata,select=-X.rotbdsmod)
  #ldata<-subset(ldata,select=-Volume)
  #ldata<-subset(ldata,select=-Area)
  if (testSet==F) { 
    cs<-colSums(abs(ldata)==0)
    cat(cs)
    if (0 %in% cs) {
      ldata<-ldata[,which(colSums((ldata))!=0)]
    }
    
  } 
  #ldata<-subset(ldata,select=-Tmult)
  #ldata<-subset(ldata,select=-shape) 
  #normalize
  return(list(ldata,smiles))
  #return(ldata)
}

prepareData_cdk<-function(ldata,lowT,highT,testSet) {
  ldata<-ldata[ldata$mpK<highT,]
  ldata<-ldata[ldata$mpK>lowT,]
  cat(nrow(ldata))
  if (testSet==T) {
    ldata<-ldata[ldata$Alkane>0,]
    #ldata<-ldata[ldata$Fragments<2,]
  } 
  smiles<-ldata[,3:4]
  ldata<-ldata[,7:length(ldata)]
  print(summary(ldata))
  if (testSet==F) { 
    cs<-colSums(abs(ldata)==0)
    if (0 %in% cs) {
      ldata<-ldata[,which(colSums((ldata))!=0)]
    }    
  } 
  #ldata<-subset(ldata,select=-Tmult)
  #ldata<-subset(ldata,select=-shape) 
  
  #normalize
  return(list(ldata,smiles))
  
}

oinfo<-function(O) {
  cat("#class:",class(O))
  cat(" #dimension:",dim(O)," ")
  cat(" #length:",length(O)," ")
  cat(" #size",object.size(O)/1000000," MB\n")
}

vinfo<-function(v) {
  cat("#class:",class(v))
  cat(" #length:",length(v)," ")
  cat(" #size",object.size(v)/1000000," MB\n")
  
}

normalizeData<-function(lX) {
  fun <- function(x){ 
    a <- min(x) 
    b <- max(x) 
    (x - a)/(b - a) 
  } 
  lX<-apply(lX, 2, fun)   
  return(lX)
}

removeZeroVars<-function(ldata) {
    cs<-colSums(abs(ldata)==0)
    if (0 %in% cs) {
      ldata<-ldata[,which(colSums((ldata))!=0)]
    }    
  return(ldata)
}


removeColVar<-function(ldata,cvalue) {
  library(caret)
  cormat<-cor(ldata)
  #print(cormat)
  c<-findCorrelation(cormat, cutoff = cvalue, verbose = TRUE)
  removed<-colnames(ldata[,c])
  cat("Removed variables according to cutoff: ",cvalue," :")
  cat(removed,"\n")
  #ldata<-subset(ldata,select=-(removed))
  ldata<-ldata[,!(names(ldata) %in% removed)]
  return(ldata)
}




linRegPredict<-function(fit,lX_test,exp,lid_test=NULL) {
  if (!is.null(lid_test)) {
    print(lid_test$SMILES)
  }
  pred<-predict(fit,lX_test) 
  plot(pred,exp,col = "blue")
  abline(0,1, col = "black")
  se<-(pred-exp)^2
  #cat("LINEAR MODEL TEST RMSE:",compRMSE(pred,exp),"\n")
  if (is.null(lid_test)) {
    predout<-data.frame(predicted=pred,exp=exp,se)
  } else {
    predout<-data.frame(id=lid_test$SMILES,predicted=pred,exp=exp,se=se)
  }  
  #print(predout)
  predout<-predout[with(predout, order(-se)), ]
  #print(predout)
  write.table(predout,file="pred_test.csv",sep=";",row.names=FALSE)
  
  return(pred)
}


linRegTrain<-function(lX,ly,lid=NULL,plot=T) {
  ldata=data.frame(lX,target=ly) 
  fit <- lm(target ~ ., data=ldata)
  #fit<-lm(mpK ~ Ringbonds + Alkylatoms + Conjugated.bonds + X.rotbdsmod + hb_self + Area + E_dielec +nbr11, data=ldata)
  if (plot==T) {
    print(summary(fit))
    plot(fit$fitted.values,ly,col="blue",pch=1, xlab = "predicted", ylab = "exp")
    abline(0,1, col = "black")
    #points(pred,ly_test,col="red",pch=2)
    t<-paste("QSPR wtih ",nrow(lX)," data points and ",length(fit$coefficients)-1, " variables.")
    title(main = t)
    # diagnostic plots 
    #layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
    #plot(fit)
    rmse<-compRMSE(fit$fitted.values,ly)
    cat("MLR TRAINING RMSE:",rmse,"\n")
    er<-(ly-fit$fitted.values)
    se<-(er)^2
    plot(ly,er,col="blue",pch=1, xlab = "target", ylab = "residual")
    #points(se,col="red",pch=1)  
    if (!is.null(lid)) {
	pred<-data.frame(lid,predicted=fit$fitted.values,exp=ly,se,er)
    } else {
	pred<-data.frame(predicted=fit$fitted.values,exp=ly,se,er)
    }
    #pred<-pred[with(pred, order(-se)), ]
    write.table(pred,file="prediction.csv",sep=",",row.names=FALSE)
    write.table(pred,file="prediction.csv",sep=";",row.names=FALSE)
    print(colnames(lX))
  }  
  return(fit)
}

varSelectGA<-function(lX,ly) {
  require(subselect)
  ldata<-data.frame(lX,mpK=ly)
  #gamodel<-genetic(cor(ldata),kmin=20,kmax=30,popsize=50,nger=1000)
  gamodel<-anneal(mat = cor(ldata), kmin = 20, kmax = 30, nsol = 50, niter = 1000,
                  criterion = "rv")
  #gamodel<-eleaps(mat = cor(ldata), kmin = 20, kmax = 30, nsol = 4, 
  #                criterion = "rv")
  #print(summary(gamodel))
  print(gamodel)
  str(gamodel)
  vars<-gamodel$bestsets[1,]
  bestdata<-data.frame(ldata[,vars])
  print(colnames(bestdata))
  #ldata<-data.frame(bestdata,mpK=ly)
  return(bestdata)
  #str(gamodel$subsets)
}

splitDataFrame<-function(df,name_list) {
  for (rname in name_list) {
    cat("name:",rname,"\n")
  }
  #print(summary(df))
  #oinfo(df)
  #row.names(df)<-df$API
  #paste(name_list,collapse="|")
  
  df_test<-df[-grep(paste(name_list,collapse="|"),df$API,invert=T),]#test set
  df_train<-df[grep(paste(name_list,collapse="|"),df$API,invert=T),]#train set
  #df_test = df[row.names(df)%in%name_list,]
  #df_train = df[!row.names(df)%in%name_list,]
  oinfo(df_test)
  oinfo(df_train)
  return(list(df_train,df_test))
}


variableSelection<-function(lX,ly,mode="forward",nvariables=5,plotting=F) {
  ldata<-data.frame(lX,target=ly)
  # All Subsets Regression
  library(leaps)
  #subsets<-regsubsets(target~.,data=ldata,nbest=1,nvmax=12,method="exhaustive",force.in=match("Tmult",names(ldata))) 
  nvariables<-nvariables
  subsets<-regsubsets(target~.,data=ldata,nbest=1,nvmax=nvariables,method=mode,force.in=NULL)
  
  
  # view results 
  final<-summary(subsets)
  #str(final)
  i <-which(final$rss==min(final$rss))
  vars <- which(final$which[i,])  # id variables of best model
  #remove intercept
  vars<-vars[2:length(vars)]-1
  bestdata<-data.frame(ldata[,vars])
  #print in reusable format e.g. python
  if (plotting==T) {
      # plot a table of models showing variables in each model. models are ordered by the selection statistic.
      plot(subsets,scale="r2")
      cat("\n[")
      for (i in 1:ncol(bestdata)) {
	cat("\"",names(bestdata)[i],"\"",sep="")
	if (i!=ncol(bestdata)) {
	    cat(",",sep="")
	  } else {
	    cat("]\n\n")
	  }
      }
  }
  
  
  # plot statistic by subset size 
  #library(car)
  #subsets(leaps, statistic="rs2")
  return(bestdata)
}

genLinMod<-function(lX,ly) {
  require(lars)
  larsmodel<-cv.lars(data.matrix(lX),data.matrix(y),plot.it=T,se=T,trace=TRUE,max.steps=80)
  str(larsmodel)
  cat("Least Angle regression RMSE:",sqrt(larsmodel$cv[100]),"\n")
  #print(summary(larsmodel))
}

cleanFrame<-function(df) {
  #remove numeric
  nums <- sapply(df, is.numeric)
  df<-df[,nums]
  #remove zero columns
  cs<-colSums(abs(df)==0)
  if (0 %in% cs) {
    df<-df[,which(colSums((df))!=0)]
  } 
  return(df)
}


compareViaPrinComp<-function(lX1,lX2=NULL,ly=NULL,threshold=0.5,plot=TRUE) {
  par(xpd=TRUE) # plot outside for legend
  #clean data: remove non-numeric & zero columns  
  lX1$train=1
  if (!is.null(lX2)) {
    lX2$train=0
    lX2<-matchByColumns(lX1,lX2)
    lX1<-data.frame(rbind(lX1,lX2))   
  }
  
  l=unique(c(as.character(lX1$API)))
  target<-droplevels(factor(lX1$API, levels=l))  
  API_ID<-as.numeric(factor(lX1$API, levels=l))  
  lX1<-cleanFrame(lX1)
  cat("COLUMNS:",colnames(lX1),"\n")
  #using ly for plotting
  if (!is.null(ly)) {    
    cat("Using label data...\n")
    #lX1<-lX1[lX1$NRing<7,]
    if (!is.null(lX2)) {
      CLASS<- droplevels(as.factor(lX1$train))
    } else {
      all_data<-data.frame(lX1,target=ly)
      CLASS<- droplevels(as.factor(all_data$target))
    }  
    lX1 <- subset(lX1, select = -train)
    lX1 <- subset(lX1, select = -solvate)
    pca <- prcomp(lX1, center=TRUE, scale.=TRUE)
    pc <- c(1,2)
    PCH<-c(15,16)[as.numeric(CLASS)]
    plot(pca$x[,pc[1]], pca$x[,pc[2]], col=CLASS,pch=PCH,xlab=paste0("PC ", pc[1], " (", round(pca$sdev[pc[1]]/sum(pca$sdev)*100,0), "%)"), ylab=paste0("PC ", pc[2], " (", round(pca$sdev[pc[2]]/sum(pca$sdev)*100,0), "%)"))
    legend_text <- c("train","validation")
    legend("bottomright", inset=c(+0.05,0.05), legend=legend_text, col=c(2,1),pch=c(16,15))
    print(summary(pca))
    #biplot(pca)
    #special for solvates
    plot(lX1$H_ex_FINE,lX1$ova_idx,col=CLASS,pch=PCH,xlab='excess enthapy',ylab='ovality')
    legend_text <- c("solvate","non-solvate")
    legend("topright", inset=c(+0.05,0.05), legend=legend_text, col=c(2,1),pch=c(16,15))
    
    #plot KDE
    require(MASS)
    lims <- c(-1.5,1.0,1.0,1.2)
    xlim<-c(-1.5,1.0)
    ylim<-c(1.02,1.2)
     
    #linear decision boundary
    #intercept_orig<-16.770
    #c_ovality<--16.596
    intercept_orig<-15.947
    c_ovality<--14.554
    c_Hex<--1.0    
    x<-seq(-1.5,1.0,0.1)
    y<--1.0*c_Hex/c_ovality*x -1.0*intercept_orig/c_ovality
    ylim_idx<- (y>1.02 & y<1.2)
    dens <- kde2d(lX1[CLASS==0,]$H_ex_FINE,lX1[CLASS==0,]$ova_idx,n = 100,lims = lims )
    image(dens,xlab = "Hex/RT",ylab="ovality",xlim=xlim,ylim=ylim)
    lines(x[ylim_idx],y[ylim_idx],col="blue")
    ylim(1.0,1.2)
    dens <- kde2d(lX1[CLASS==1,]$H_ex_FINE,lX1[CLASS==1,]$ova_idx,n = 100,lims = lims)
    image(dens,xlab = "Hex/RT",ylab="ovality",xlim=xlim,ylim=ylim)
    lines(x[ylim_idx],y[ylim_idx],col="blue")
    ylim(1.0,1.2)
    
    
    #points(x,y)
    
    #contour(dens, add=T)
    
    print(summary(pca))
    
  } else {
    lX1_tmp <- subset(lX1, select = -train) 
    pca <- prcomp(lX1_tmp, center=TRUE, scale.=TRUE)#prcomp is via SVD, princomp via eigen value decomp
    print(summary(pca)) # print variance accounted for
    print(pca)
    if (plot) {
      COLOR <- c(1:length(target))   
      TRAIN<- droplevels(as.factor(lX1$train))    
      pc <- c(1,2)
      plot(pca, type = "l")
      plot(pca$x[,pc[1]], pca$x[,pc[2]], col=COLOR[target], pch=API_ID, xlab=paste0("PC ", pc[1], " (", round(pca$sdev[pc[1]]/sum(pca$sdev)*100,0), "%)"), ylab=paste0("PC ", pc[2], " (", round(pca$sdev[pc[2]]/sum(pca$sdev)*100,0), "%)"),xlim=c(-4,4))
      #plot(pca$x[,pc[1]], pca$x[,pc[2]], col=COLOR[target], pch=as.character(lX1$API_ID), xlab=paste0("PC ", pc[1], " (", round(pca$sdev[pc[1]]/sum(pca$sdev)*100,0), "%)"), ylab=paste0("PC ", pc[2], " (", round(pca$sdev[pc[2]]/sum(pca$sdev)*100,0), "%)"))
      #text(pca$x[,pc[1]], pca$x[,pc[2]],label=as.character(lX1$API_ID),col=COLOR[target])      
      legend_text <- c(1:length(API_ID))
      legend("topright", inset=c(-0.05,0), legend=levels(target), col=COLOR,pch=legend_text)
      #legend("topleft", legend=c("test", "train"), col=1, pch=PCH)
      #biplot(pca)
      #http://stats.stackexchange.com/questions/72839/how-to-use-r-prcomp-results-for-prediction
    }
    #print(summary(pca$x))
  }
  
}


pc_analysis<-function(lX,ly,lX_test=NULL) {
  #mydata<-data.frame(lX,target=ly)
  mydata<-data.frame(lX)
  if (!is.null(lX_test)) {
    mydata<-data.frame(rbind(lX,lX_test))
  }
  #remove zero columns
  cs<-colSums(abs(mydata)==0)
  #cat(cs)
  if (0 %in% cs) {
    mydata<-mydata[,which(colSums((mydata))!=0)]
  } 
  fit <- princomp(mydata, cor=TRUE)
  print(summary(fit)) # print variance accounted for 
  #print(loadings(fit)) # pc loadings 
  plot(fit,type="lines") # scree plot 
  #print(fit$scores) # the principal components
  #labels <- 1:nrow(lX)
  labels <- rep("-",nrow(lX))
  biplot(fit,xlabs=labels)
}

matchByColumns<-function(df_orig,df_test) {
  validCols<-colnames(df_orig)
  cat("Keeping variables:",validCols,"\n")
  df_test<-subset(df_test,select=validCols)
  return(df_test)
}

compRMSE<-function(a,b) {
  mse<-sum((a-b)^2,na.rm=TRUE)
  rmse<-sqrt(mse/sum(!is.na(a)))
  #cat("RMSE:",rmse)
  return(rmse)
}

compAARD<-function(a,b) {
  aard<-sum((abs(a-b)/b),na.rm=TRUE)
  aard<-aard/sum(!is.na(a))
  return(aard)
}

compLOGLOSS<-function(predicted,actual) {
  eps<-1e-15
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  result<- -1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}

compRSQ<-function(predicted,actual) {
   rsq<-cor(predicted,actual,method="pearson")
   return(rsq**2)
}


gam_model<-function(lX,ly,verbose) {
  library(mgcv)
  mydata<-data.frame(lX,target=ly)
  #gam1<-gam(mpK ~ s(avratio) + s(Rotatable.bonds)+s(Ringbonds)+s(hb_self)+s(E_dielec)+ s(Txentropy)+s(X.rotbdsmod_new),data=mydata)
  #gam1<-gam(mpK ~ s(avratio) + s(Txentropy)+s(Ringbonds)+s(Conjugated.bonds)+s(hb_self_mod)+s(E_dielec)+ s(HB.don_m2)+s(other_rotbonds)+s(n_total),data=mydata)
  #best lin model
  #gam1<-gam(target ~ s(avratio)+s(hb_self_lowT) + s(Molweight..g.mol.) +s(Conjugated.bonds) + s(HB.don_m1) +s(E_dielec)+s(Rotatable.bonds)+s(nbr11)+s(Kier2),data=mydata)
  #gam1<-gam(target ~ Hex.kcal.mol.+s(ringbonds_API,k=3)+ringbonds2+ln_vcosmo+s(M2_API,k=3),data=mydata,family=binomial(link=logit),method="GCV.Cp")
  gam1<-gam(target ~ Hex.kcal.mol.+s(ringbonds_API,k=3)+ringbonds2+ln_vcosmo+s(ln_vfree_drug,k=3),data=mydata,family=binomial(link=logit),method="GCV.Cp")
  
  #gam1<-gam(target ~ Hex.kcal.mol.+s(ln_vcosmo,k=3),data=mydata,family=binomial(link=logit),method="GCV.Cp")
  #gam1<-gam(mpK ~ s(avratio,k=4) + s(Txentropy)+s(Conjugated.bonds)+s(hb_self_mod)+s(E_dielec),data=mydata)

  #print(summary(gam1))
  #str(gam1)  
  #cat("GAM RMSE:",compRMSE(gam1$fitted.values,ly),"\n")
  if (verbose==T) {
    plot(gam1,residuals=FALSE,pch=12,shade=T, scale=0)
    gam.check(gam1,pch=19,cex=.3)
  }
  return(gam1)
}

trainResidues<-function(lX,ly,linvalues,iter,useRF) {
  cat("Train residue model:\n")
  residues<-ly-linvalues
  hist(residues,50)
  plot(residues,ly)
  if (useRF==T) {
    model<-trainRF(lX,residues,iter)    
    residue_pred<-model$predicted
  } else {
    model<-linRegTrain(lX,residues,null,F)
    residue_pred<-model$fitted.values
  }
  finalpred<-linvalues+residue_pred
  plot(finalpred,ly,col="blue",xlab = "predicted", ylab = "exp")
  abline(0,1, col = "black")
  rmse<-compRMSE(finalpred,ly)
  cat("RMSE (lin+RF):",rmse,"\n")
  results<-data.frame(exp=ly,lin=linvalues,residues=residues,rfcorr=residue_pred,finalpred=finalpred)
  write.table(results,file="residue_train.csv",sep=",",row.names=FALSE)
  return(model)
}

predictResidues<-function(lX,ly,model,linvalues) {
  cat("Predict residue model:\n")
  residues<-predict(model,lX)
  finalpred<-linvalues+residues
  points(finalpred,ly,col="red",xlab = "predicted", ylab = "exp")
  abline(0,1, col = "black")
  cat("RMSE,test (lin+RF):",compRMSE(finalpred,ly),"\n")
  cat("R^2,test (lin+RF):",(cor(finalpred,ly))^2,"\n")
  cat("slope,test (lin+RF):",getSlope(finalpred,ly),"\n")
  results<-data.frame(exp=ly,lin=linvalues,residues_true=ly-linvalues,rfcorr=residues,finalpred=finalpred)
  write.table(results,file="residue_test.csv",sep=",",row.names=FALSE)
  #cat(residues)  
}

pc_correct<-function(actual,predicted) {
  sum<-0.0
  for(i in 1:length(actual)) {
    if (actual[i]==predicted[i]) {
      sum<-sum+1
    }
  }
  cat("Sum correct predictions:",sum,"\n")
  pc<-sum/length(actual)
  cat("% correct predictions:",pc,"\n")
  cat("% wrong predictions:",(1-pc),"\n")
  return(pc)
}


sigmoidal<-function(x,a=1.0,b=0.0) {
  value<-1/(1+exp(-a*(x-b)))
  return(value)
}

#computes enrichment factor
computeEF<-function(predicted,truth,returnMax=FALSE,top=1.0,invert=T,verbose=F) {
  if(is.factor(truth)) truth<-as.numeric(as.character(truth))
  if(invert) {
    truth<-(-1*truth+1)
    predicted<--1*predicted
  }
  res<-data.frame(pred=predicted,truth=truth)
  res<-res[with(res, order(-predicted)), ]
  Nfh<- match(1, res$truth)
  nh<-sum(res$truth)
  N<-length(res$truth)

  if (returnMax) {
    EF<-N/nh
  } else {
    EF<-(1.0/Nfh)/(nh/N)
  }
  if (verbose) {
    cat(sprintf("Number to first hit(Nfh): %4d\n",Nfh))
    cat(sprintf("Number of hits(nh): %4d\n",nh))
    cat(sprintf("Total N: %4d\n",N))
    cat(sprintf("Enrichment: %5.2f\n",EF))
  }
  
  return(EF)
}

computeMultiLogLoss <- function(act, pred,verbose=F)   {
      eps = 1e-15;
      nr <- nrow(pred)
      pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)      
      pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
      cat("pred:",pred,"\n")
      ll = sum(act*log(pred) )
      ll = ll * -1/(nrow(act)) 
      if (verbose) {
	cat("logloss:",ll,"\n")
      }
      return(ll);
    }

confintAUC<-function(auc,ytrue) {
  #see J Comput Aided Mol Des (2014) 28:887-918
  n_pos<-sum(ytrue==1)
  n_neg<-sum(ytrue==0)
  var_pos<-auc**2*(1-auc)/(1+auc)/n_pos
  var_neg<-auc*(1-auc)**2/(2-auc)/n_neg
  var<-1.96*sqrt(var_pos+var_neg)
  return(var)
}

computeROC_EF<-function(predicted,truth,cutoff=0.01) {
  #computes ROC enrichment J Comput Aided Mol Des (2014) 28:887-918
  #http://www.ub.edu/cbdd/?q=content/how-calculate-roc-curves
  #works only with smooth curves
  require("ROCR")
  pred<-prediction(predicted, truth)
  perf <- performance(pred,"tpr","fpr")
  EF_roc <- perf@y.values[[1]]/perf@x.values[[1]]
  cat("Fraction neg.:",perf@x.values[[1]],"\n")
  cat("EF_roc:",EF_roc,"\n")
  EF_roc_cut <- EF_roc[which(perf@x.values[[1]] > cutoff)[1]]
  return(EF_roc_cut) 
}

computeF1score<-function(predicted,truth) {
  retrieved <- sum(predicted)
  #Precision is the fraction of retrieved instances that are relevant
  precision <- sum(predicted & truth) / retrieved
  #Recall is the fraction of relevant instances that are retrieved
  recall <- sum(predicted & truth) / sum(truth)
  f1score <- 2 * precision * recall / (precision + recall)
  return(f1score)
}

computeAccuracy<-function(predicted,truth,threshold=0.5,metric='Accuracy',verbose=F) {
  #Computes accuracy from probabilities
  predicted<-as.numeric(predicted)#for GAM model
  predicted<- factor( ifelse(predicted < threshold, "0", "1") )
  cm<-confusionMatrix(predicted, as.factor(truth),positive = "1")
  if (verbose) print(cm)
  acc<-cm$overall[metric]
  return(acc)
}


computeAUC<-function(predicted,truth,titlename=NULL,verbose=F,predicted2=NULL) {
  require("ROCR")
  predicted<-as.numeric(predicted)#for GAM model
  pred<-prediction(predicted, truth)
  perf <- performance(pred,"tpr","fpr")

  auc<-performance(pred,"auc")
  auc<-unlist(slot(auc, "y.values"))
  if (verbose) {
    if (auc<0.70)  {
      color<-"orange"
      lty<-2
    } else if (auc<0.55)  {
        color<-"red"
        lty<-1
    } else {
      color<-"blue"
      lty<-1
    }
    plot(perf,col=color,lty=lty, lwd=3,xlim=c(0,1),xlab="FPR",ylab="TPR")
    if (!is.null(predicted2)) {
      pred2<-prediction(predicted2, truth)
      perf2 <- performance(pred2,"tpr","fpr")
      auc2<-performance(pred2,"auc")
      auc2<-unlist(slot(auc2, "y.values"))    
      lines(perf2@x.values[[1]],perf2@y.values[[1]],col=color,lty=3, lwd=3,xlim=c(0,1),xlab="FPR",ylab="TPR")
      legend("bottomright", legend=c("fit","hex"), col='blue',lty=c(1,3))
    }
    #abline(0.0,1.0, col = "black",lwd=3,xlim=c(0,1))
    lines(seq(0.0,1.0,0.1),seq(0.0,1.0,0.1),col = "black",lwd=2)
    xlim(0.0,1.0)
    ylim(0.0,1.0)
    if (!is.null(titlename)) {
      title(titlename,cex.main=1.0) # scale title
      if (!is.null(predicted2)) {
        text(0.6,0.1,sprintf("AUC: %6.2f (%3.2f)",auc,auc2))
      } else {
        text(0.6,0.1,sprintf("AUC: %6.2f",auc))
      }      
    }
  }
  auc<-performance(pred,"auc")
  auc<-unlist(slot(auc, "y.values"))
  #str(auc)
  #if (verbose) cat("AUC:",auc,"\n")
  return(auc)
}

getSlope<-function(lx,ly) {
  slopeinfo<-lm(ly~lx)
  slope<-slopeinfo$coefficients[2]
  #cat("Slope:",slope,"\n")
  return(slope)
}

#optimizing parameters for gbm
caret_train<-function(Xl,yl,classification=F,lmethod="gbm") {
  oinfo(Xl)
  #require(gbm)
  require(caret)
  #options(warn=-1)
  
  ####windows#########
  require(doSNOW)
  #cat("procs:",getDoParWorkers()," name:",getDoParName()," version:",getDoParVersion(),"\n")
  #cl<-makeCluster(nrfolds, type = "SOCK",outfile="")
  #registerDoSNOW(cl)
  
  ####Linux############
  #library(doMC)
  #registerDoMC(4)
  
  if (classification) {
    cat("classificiation\n")
    yl<-factor(yl)    
  }
  #at least 3 iter values..?
  grid<-expand.grid(.interaction.depth = c(4,5,6),.n.trees = c(5000,10000,15000),.shrinkage = c(0.01,0.05,0.001))
  #grid<-createGrid("gbm", data = tmp, len = 3)
  #print(summary(grid))
  #grid<-expand.grid(.K.prov=c(4,5))
  #grid<-expand.grid(.k=c(10,20,50))
  #grid<-expand.grid(.treesize=c(2,4))
  #fitControl <- trainControl(method = "cv",number = 5,summaryFunction = twoClassSummary,verboseIter=TRUE,classProbs=TRUE)
  #fitControl <- trainControl(method = "repeatedcv",repeats = 5,classProbs = classification,verboseIter=TRUE)
  fitControl <- trainControl(method = "repeatedcv",number = 5,repeats = 2,verboseIter=F)
  if (classification) {
    model <- train(Xl, yl,method = lmethod, trControl = fitControl,metric="ROC", tuneGrid = grid,maximize=TRUE)
    plot(model,metric="ROC",plotType="level") 
  } else {
    model <- train(Xl, yl,method = lmethod, trControl = fitControl,metric="RMSE", tuneGrid = grid,maximize=FALSE,verbose=FALSE)
    plot(model,metric="RMSE",plotType="level") 
  }
  print(model)
  print(model$results)
  #str(gbmFit1$results)
  #expred <- extractPrediction(list(gbmFit1))
  #plotObsVsPred(expred)
  resampleHist(model)
  summary(model)
  #
}

gbm_grid<-function(lX,ly,lossfn="auc",treemethod="gbm",repeatcv=3,parameters=list(depth=c(4,8),shseq=c(0.01,0.02),iterations=c(500), minobsinnode=c(5))) { 
  df<-data.frame(lX,ly) 
  iterations<-parameters[["iterations"]]#number of tress
  if (treemethod=="gbm"|| treemethod=="xgboost") { 
    param_a<-parameters[["depth"]] 
    param_b<-parameters[["shseq"]]
    param_c<-parameters[["minobsinnode"]]
  } else {
    param_a<-parameters[["mtry"]]
    param_b<-c(1)
    param_c<-parameters[["nodesize"]]
    sampsize<-parameters[["sampsize"]]
  } 
  for(i in param_a) {
    for(j in param_b) {
      for(k in iterations) {
        for(l in param_c) {
          if (treemethod=="gbm" || treemethod=="xgboost") {
            cat("Iterations:",k," int.depth:",i, " shrinkage:",j," minobsinnodee:", l)
            xval_oob(lX,ly,iterations=k,nrfolds=5,intdepth=i,sh=j,minobsinnode=l,repeatcv=repeatcv,lossfn=lossfn,method=treemethod)     
          } else {
            cat("Iterations:",k," mtry:",i," nodsize:",l,"\n")
            xval_oob(lX,ly,iterations=k,nrfolds=5,mtry=i,minobsinnode=l,repeatcv=repeatcv,lossfn=lossfn,method=treemethod)
          }
          cat("\n")
        }
      }
    }
  }  
}

#parallel crossvalidation that delivers also out-of-bag predictions 
xval_oob<-function(Xl,yl,Xtest=NULL,iterations=500,nrfolds=5,intdepth=2,sh=0.01,minobsinnode=10,repeatcv=2,lossfn="rmse",method="gbm",mtry=5,oobfile='oob_res.csv',folds_from_file=NULL,analyzeResidues=FALSE,verbose=F) {
  if (verbose) cat("\nxval-oob - repeats: ",repeatcv)
  if (!is.null(folds_from_file)) {
    cat(" - folds from file:",folds_from_file,"\n")
    saveFoldsToFile(Xl,k=nrfolds,n=repeatcv,filename="folds.csv")
  } 
  
  #outer serial loop, column bind!!!
  results_all<-foreach(j = 1:repeatcv,.combine="cbind") %do% {      
    if (!is.null(folds_from_file)) {
      cat("Reading folds from file, pos:",j,"\n")
      train<-createFoldsFromFile("folds.csv",pos=j)
    }  else {
      train<-createFoldIndices(Xl,k=nrfolds)
    }    
    cl<-makeCluster(nrfolds, type = "SOCK")
    registerDoSNOW(cl) 
    
    #inner parallel loop, automatically rbind results to res use dopar
    res_all<-foreach(i = 1:nrfolds,.packages=c('gbm','randomForest','xgboost'),.combine="rbind",.export=c("compRMSE","computeAUC","oinfo","linRegTrain","variableSelection")) %do% { 
      if (verbose) cat("###FOLD: ",i,"\n") 
      idx <- which(train == i)
      Xtrain<-Xl[-idx,,drop=F]
      ytrain<-yl[-idx]
      Xvalid<-Xl[idx,,drop=F]
      ytest<-yl[idx]
      test_data<-NULL
      #CLASSIFICATION
      if (lossfn=="auc") {
        if (method=="gbm") {
          #GBM
          gbm1<-gbm.fit(Xtrain ,ytrain,distribution="bernoulli",n.trees=iterations,interaction.depth=intdepth,shrinkage=sh,n.minobsinnode = minobsinnode,verbose=F)
          oob_pred<-predict.gbm(gbm1,Xvalid,n.trees=iterations,type="response")
          if (!is.null(Xtest)) {
            test_pred<-predict.gbm(gbm1,Xtest,n.trees=iterations,type="response")
          }
        } else if (method=='randomForest' || method=='rf') {
          #RF
          rf1 <- randomForest(Xtrain,ytrain,ntree=iterations,mtry=mtry,nodesize=minobsinnode,importance = F)
          oob_pred<-predict(rf1,Xvalid,type="vote")[,2]
          if (!is.null(Xtest)) {
            test_pred<-predict(rf1,Xtest,type="vote")[,2]
          }
        } else if (method=='linear') {
          #linear regression 
          Xtrain$ytrain <- ytrain          
          fit<-glm(ytrain~., data=Xtrain,family=binomial(link="logit"))
          oob_pred<-predict(fit,Xvalid,type='response')
          if (!is.null(Xtest)) {
            test_pred<-predict(fit,Xvalid)
          }
        }
        score<-computeAUC(oob_pred,ytest,F)
      #REGRESSION
      } else {
        if (method=="gbm") {
          #GBM
          gbm1<-gbm.fit(Xtrain ,ytrain,distribution="gaussian",n.trees=iterations,interaction.depth=intdepth,shrinkage=sh,n.minobsinnode = minsnode,verbose=F)
          oob_pred<-predict.gbm(gbm1,Xvalid,n.trees=iterations,type="response")
          if (!is.null(Xtest)) {
            test_pred<-predict.gbm(gbm1,Xtest,n.trees=iterations,type="response")
          }
        } else if (method=='xgboost'|| method=='xgb') {
          xgb_params <- list(objective = "reg:linear",
                             eval_metric = "rmse",
                             max.depth=intdepth,
                             eta=sh,
                             silent=1, 
                             subsample=0.5,
                             nthread = 1,
                             min_child_weight = minobsinnode)
           
          fit<-xgboost(param=xgb_params, data =as.matrix(Xtrain), label= ytrain,nrounds=iterations,missing = NA)
          oob_pred <-predict(fit,as.matrix(Xvalid),outputmargin=T,predleaf=F, missing = NA)#XGBOOST
          if (!is.null(Xtest)) {
            test_pred<-predict(fit,as.matrix(Xtest),outputmargin=T,predleaf=F, missing = NA)#XGBOOST
          }         
        } else if (method=='randomForest'|| method=='rf') {
          #RF
          rf1 <- randomForest(Xtrain,ytrain,ntree=iterations,mtry=mtry,importance = F)
          oob_pred<-predict(rf1,Xvalid)
          if (!is.null(Xtest)) {
            test_pred<-predict(rf1,Xtest)
          }
        } else if (method=='linear') {
          #linear regression
          Xtrain<-variableSelection(Xtrain,ytrain,"forward",iterations)
          fit<-linRegTrain(Xtrain,ytrain,NULL,F)
          oob_pred<-predict(fit,Xvalid)
          if (!is.null(Xtest)) {
            test_pred<-predict(fit,Xtest)
          }
        }
        score<-compRMSE(oob_pred,ytest)
      }        
      if (verbose) cat(" LOSS:",score,"\n") 
     
      #returning dataframe with predictions and truth
      oob_data<-data.frame(idx,pred=oob_pred,ytest=ytest,testidx=-1)     
      if (!is.null(Xtest)) {
        test_data<-data.frame(idx=-1,pred=test_pred,ytest=-1,testidx=seq(1,length(test_pred)))
      }      
      #automatic rbin for each iteration!!
      oob_data<-rbind(oob_data,test_data)
      return(oob_data)
    }#end parallel inner loop
    #average test set prediction via fold index
    if (!is.null(Xtest)) {
      pred_df<-res_all[which(res_all$idx<0),]
      pred_df<-aggregate(pred_df$pred, by=list(pred_df$testidx),FUN=mean)[,2]
    }
    res<-res_all[which(res_all$idx>0),]
    #restore original order 
    res<-res[order(res$idx),]   
    if (lossfn=="auc") {
      auc_cv<-computeAUC(res$pred,res$ytest)
      if (verbose) cat("AUC, CV:",auc_cv,"\n")
    } else {
      auc_cv<-compRMSE(res$pred,res$ytest)
      if (verbose) cat("RMSE, CV:",auc_cv,"\n")
    }
    #tmp<-c(res$pred,pred_df)
    tmp<-c(res$pred)
    stopCluster(cl) 
    return(tmp)
  }#end outer loop

  mean_all<-apply(results_all, 1, function(x) mean(x))
  results_oob<-results_all[1:nrow(Xl),]
  mean_oob<-mean_all[1:nrow(Xl)]
  
  if (lossfn=="auc") {
    score_iter<-apply(results_oob, 2, function(x) computeAUC(x,yl,F))
    if (verbose) cat("AUC of iterations:",score_iter,"\n")
    if (verbose) cat("<AUC,oob>:",computeAUC(mean_oob,yl,F),"\n") 
  } else {
    score_iter<-apply(results_oob, 2, function(x) compRMSE(x,yl))
    cat("RMSE of iterations:",score_iter,"\n")
    cat("<RMSE,oob>:",compRMSE(mean_oob,yl)," RMSE,mean:",mean(score_iter)," sdev:",sd(score_iter),"\n") 
  }
  #write oob and testset prediction to file
  res<-data.frame(prediction=mean_all)
  if (!is.null(oobfile)) {
    write.table(res,file=oobfile,sep=",",row.names=FALSE)
  }
  #look at oob predictions
  if (analyzeResidues) {
    plot(mean_oob,yl,col = "blue")
    abline(0,1, col = "black")
    residue = yl- mean_oob
    plot(mean_oob,residue,col = "red")
    write.table(data.frame(prediction=mean_oob,exp=yl,residue=residue),file="oob_only.csv",sep=",",row.names=FALSE)
  }
  return(res)
}



trainRF<-function(lX,ly,iter=500,mtry=if (!is.null(ly) && !is.factor(ly)) max(floor(ncol(lX)/3), 1) else floor(sqrt(ncol(lX))),node.size=5, verbose=T,fimportance=F) {
  require(randomForest)
  cat("Training random forest...") 
  mydata.rf <- randomForest(lX,ly,ntree=iter,mtry=mtry,importance = fimportance,nodesize =node.size)
  
  print(mydata.rf)
  print(summary(mydata.rf))
  if (fimportance) {
    imp<-importance(mydata.rf,type=1)
    varImpPlot(mydata.rf,type=1,main="")
    #write.table(data.frame(imp),file="importance.csv",sep=",")
    write.table(data.frame(imp),file="importance.csv",sep=",")
    #png(file="importance.png",width=1600,height=1200,res=300)
    #par(mar=c(4,1,2,2))
    #dev.off()
  }

  if (mydata.rf$type=="regression") {  
    cat("...regression\n")
    nr.samples<-nrow(lX)
    rmse<-mean(sqrt(mydata.rf$mse)*(nr.samples-1)/nr.samples)
    stdev<-sd(sqrt(mydata.rf$mse))
    if (verbose==T) {     
      cat("RF RMSE:",rmse,"stddev:",stdev," ")
      cat("R^2:",mean(mydata.rf$rsq),"\n")
      #plot
      results<-data.frame(predicted=mydata.rf$predicted,exp=mydata.rf$y)
      plot(results,col="blue",pch=1, xlab = "predicted", ylab = "exp")
      abline(0,1, col = "black")
      t<-paste("RF wtih ",nrow(lX)," data points and ",ncol(lX), " variables.")
      t<-paste("RF with ",nrow(lX)," data points and ",ncol(lX), " variables.")
      title(main = t) 
      #residues: y-pred 
      results<-data.frame(predicted=mydata.rf$predicted,residue=mydata.rf$y-mydata.rf$predicted)
      plot(results,col="blue",pch=1, xlab = "exp", ylab = "residues")
      title(main = "RF residuals (y-pred)")
      se<-(mydata.rf$predicted-mydata.rf$y)^2
      ldata<-data.frame(predicted=mydata.rf$predicted,exp=mydata.rf$y,se,residue=mydata.rf$y-mydata.rf$predicted)
      #if (!is.null(rownames)) {
      #  ldata<-data.frame(name=rownames,predicted=mydata.rf$predicted,exp=mydata.rf$y,se)
      #}
      
    }      
  } else if (length(mydata.rf$classes)>2) {
    cat(" multi classification:",mydata.rf$classes,"\n")
    #cat(mydata.rf$votes)
    #ly is factor
    ly<-as.numeric(as.integer(ly))
    str(ly)
    score<-computeMultiLogLoss(ly,mydata.rf$votes,T)
    cat(score)
    
  } else {
    cat(" classification\n")
    if (verbose==T) {
      print(nlevels(ly))
      cat("Random forest OOB err rate:",mydata.rf$err.rate[iter],"\n")
      #tmp<-mydata.rf$predicted
      tmp<-mydata.rf$votes[,2]   
      #hist(as.numeric(mydata.rf$predicted))
      hist(as.numeric(tmp))
      #ldata<-data.frame(predicted=as.numeric(as.character(tmp)),truth=as.numeric(as.character(ly)))      
      ldata<-data.frame(predicted=tmp,truth=ly)
      computeAUC(ldata$predicted,ldata$truth,T)
      #pc_correct(ldata$predicted,ldata$truth)
    }
  }   
  write.table(ldata,file="prediction_rf.csv",sep=",",row.names=FALSE)
  #write.table(ldata,file="prediction_rf.csv",sep=";",row.names=FALSE)
  return(mydata.rf)
}

saveModel<-function(lX,ly,iter) {
  require("pmml")
  require("XML")
  require("randomForest")
  mydata<-data.frame(lX,mpK=ly)
  #load("mydata_rf.RData")
  print(summary(mydata))
  #print(summary(lX))
  #model<-randomForest(Sepal.Length ~ ., data=iris)
  cat("Training Forest\n")
  model<-randomForest(mpK ~ .,data=mydata,ntree=iter,importance = FALSE,nodesize=5)
  
  nr.samples<-nrow(lX)
  rmse<-mean(sqrt(model$mse)*(nr.samples-1)/nr.samples)
  stdev<-sd(model$mse)
  cat("RMSE:",rmse,"stddev:",stdev,"\n")
  cat("R^2:",mean(model$rsq),"\n")
  
  write.table(data.frame(mydata,predicted=model$predicted),file="simple_data.csv",sep=",",row.names=FALSE)
  #mytree<-getTree(model, k=1, labelVar=FALSE)
  #str(mytree)
  print(model)
  #EXPORT ONLY WORKS FOR RF CREATED WITH FORMULA INTERFACE?
  cat("Exporting PMML\n")
  p<-pmml(model)
  saveXML(p,"model.xml")
  return(model)
}


mlr_analysis<-function(fit,lX,lid,lname) {
  n<-which(lid$SMILES==lname)
  cat("index",n," compound: ",toString(lid[n,2:2]),"\n")
  obs<-fit$model[n,]
  sum<-0.0
  cat(sprintf("%-24s%12s%12s%12s%12s\n","Variable","Value","Factor", "Incr.","Total"))
  for(i in 2:length(fit$coefficients)) {
    contrib<-obs[1,i]*fit$coefficients[i]
    sum<-sum+contrib
    if (obs[1,i]<10e-15 && obs[1,i]>-10e-15) {
      next
    }
    cat(sprintf("%-24s",names(fit$coefficients)[i]))    
    cat(sprintf("%12.2f",obs[1,i]))
    cat(sprintf("%12.2f",fit$coefficients[i]))    
    cat(sprintf("%12.2f %12.2f\n",contrib,sum))
  }
  sum<-sum+fit$coefficients[1]
  cat("Intercept:",fit$coefficients[1])
  #str(fit$model)
  cat("\n##Calc. Property:",(sum)," true: ",fit$model[n,"target"], "residual: ",fit$residuals[n] ,"##\n")
}

prepareStandard<-function(filename) {
  ldata = read.csv(filename)
  ldata<-ldata[,3:length(ldata)]  
  cs<-colSums(abs(ldata)==0)
  if (0 %in% cs) {
    ldata<-ldata[,which(colSums((ldata))!=0)]
  }
  #print(summary(ldata))
  return(ldata)
}




#works not properly loss function too low -> use xvalid instead
xvalSVM<-function(lX,ly) {
  require(ipred)
  require(e1071)
  mydata<-data.frame(lX,target=ly)
  error.SVM <- numeric(10)
  for (i in 1:5) error.SVM[i] <-errorest(target~.,data=mydata,model = svm, cost = 10, gamma = 1.5)$error
  print(summary(error.SVM))
}

trainSVM<-function(lX,ly) {
  require(e1071)
  mydata<-data.frame(lX,target=ly)
  #mydata<-data.frame(lX,target=ly)
  m <- svm(target~ .,data=mydata,cost = 10, gamma = 1.5)
  new <- predict(m, lX)
  plot(ly, new)
  rmse<-compRMSE(ly,new)
  cat("RMSE:",rmse,"\n")
  cor.p<-cor(ly,new)
  cat("R^2:",cor.p,"\n")
  #cat("R^2:",mean(mydata.rf$rsq),"\n")  
}

#Local outlier Factor
#http://www.rdatamining.com/examples/outlier-detection
#http://www.dbs.ifi.lmu.de/~zimek/publications/KDD2010/kdd10-outlier-tutorial.pdf
outlierDetection<-function(lX,ly,nrPCA,sortOnResidues=FALSE,plot=TRUE,lfit=NULL,lsmiles=NULL,returnAll=FALSE) {
  require(DMwR)
  nout=20
  nneighbors=5
  cat("Starting outlier detection...")
  outlier.scores <- lofactor(lX, k=nneighbors)
  str(outlier.scores)
  plot(density(outlier.scores))
  
  if (sortOnResidues==TRUE) {
    res<-lfit$residuals
    outliers <- order(res, decreasing=T)[1:nout]
    cat("correlation res-outlier_scores:",cor(res,outlier.scores),"\n")
  } else {
    outliers <- order(outlier.scores, decreasing=T)[1:nout]
  }
  print(summary(outlier.scores))
  n <- nrow(lX)
  labels <- 1:n
  labels[-outliers] <- "."
  biplot(prcomp(lX), cex=.8, xlabs=labels)
  if (!is.null(lsmiles)) {
    predout<-data.frame(lX[outliers,],lsmiles[outliers,],outl_score=outlier.scores[outliers],exp=ly[outliers],predicted=lfit$fitted.values[outliers],res=res[outliers])
    write.table(predout,file="outlier.csv",sep=";",row.names=FALSE)
  }
  #write test set with labels
  
  if (returnAll==T) {
    newdf<-data.frame(lX,outl_score=outlier.scores)
  } else {
    newdf<-data.frame(outl_score=outlier.scores)
  }
  
  return(newdf)
}


#model:rf,iter,gam,svm,boost,gbm
xvalid<-function(lX,ly,nrfolds=5,modname="rf",lossfn="auc",parameters=list(iter=20000,depth=10,shseq=0.005,minobsinnode=5,gbm_steps=100,mtry=5)) {
  require(e1071)
  require(gbm)
  ldata=data.frame(lX,target=ly)
  all_folds<-random_folds(ldata,k=nrfolds)  
  loss<-mat.or.vec(nrfolds,1)
  for(i in 1:nrfolds) {
    train<-return_fold(all_folds,i,test=F)
    Xtrain<-train[,1:ncol(train)-1]
    ytrain<-train[,length(train)]
    test<-return_fold(all_folds,i,test=T)
    Xtest<-test[,1:ncol(test)-1]
    ytest<-test[,length(test)]
    ############################################# 
    # Random Forest                             #
    #############################################
    if (modname=="rf") {
      fit<-trainRF(Xtrain,ytrain,parameters$iter,mtry=parameters$mtry,verbose=T)
    }
    else if (modname=="gam") {
      fit<-gam_model(Xtrain,ytrain,F)
    }
    ############################################# 
    #SVM                                        #
    #############################################
    else if (modname=="svm") {
      cat("Training SVM\n")
      #For some reason we have to use the formula interface
      #traindata=data.frame(Xtrain,target=ytrain)
      #fit<-svm(target ~ .,data=traindata)
      fit<-svm(Xtrain,ytrain, kernel='radial',probability=T)
    #############################################
    #  LINEAR MODEL + RF                        #
    #############################################
    } else if (modname=="boost") {
      cat("TRAIN LINMODEL+RF: ")
      fit<-linRegTrain(Xtrain,ytrain,NULL,F)
      residues<-ytrain-fit$fitted.values
      rfmodel<-trainRF(Xtrain,residues,parameters$iter)
      print(rfmodel)
      finalpred<-fit$fitted.values+rfmodel$predicted
      if (lossfn=="rmse") {
        loss<-compRMSE(finalpred,ytrain)
        cat("RMSE:",rmse,"\n")
      } else {
        loss<-computeAUC(finalpred,ytrain,F)
        cat("AUC:",loss,"\n")
      } 
    ############################################# 
    #GRADIENT BOOSTING WITH EARLY STOPPING      #
    #############################################    
    } else if (modname=="gbm") {    
      gbm_steps<-parameters$gbm_steps
      cat("TRAIN GBM with steps: ",gbm_steps," - ")
      iter_local<-parameters$iter/gbm_steps
      cat(sprintf("ITERATIONS: %6d DEPTH: %6d SHRINkAGE: %6f MINOBS_IN_NODE: %3d",parameters$iter,parameters$depth,parameters$shseq,parameters$minobsinnode))    
      fits<- vector(mode = "list", length = gbm_steps)
      loss_test<-mat.or.vec(gbm_steps,1)
      loss_train<-mat.or.vec(gbm_steps,1)
      for (j in 1:gbm_steps) {     
        if (j==1) {
          if (lossfn=="rmse") {
            cat("REGRESSION...\n")
            fits[[j]]<-gbm.fit(Xtrain,ytrain,distribution="gaussian",n.trees=iter_local,interaction.depth=parameters$depth,n.minobsinnode=parameters$minobsinnode,shrinkage=parameters$shseq,verbose=F)         
            } else {
            cat("0-1 distribution...\n")
            fits[[j]]<-gbm.fit(Xtrain,ytrain,distribution="bernoulli",n.trees=iter_local,interaction.depth=parameters$depth,n.minobsinnode=parameters$minobsinnode,shrinkage=parameters$shseq,verbose=F)
          }        
        } else {  
          fits[[j]]<-gbm.more(fits[[j-1]],iter_local,verbose=F)
          inbag<-predict(fits[[j]],n.trees=iter_local*j,Xtrain,type="response")          
          oobag<-predict(fits[[j]],n.trees=iter_local*j,Xtest,type="response")
          loss_train[j]<-compRMSE(inbag,ytrain)
          loss_test[j]<-compRMSE(oobag,ytest)
          #cat("iter: ",j,"RMSE,train:",loss_train[j],"  RMSE,test:",loss_test[j]," \n")
          cat(sprintf("iter: %6d RMSE,train: %6.2f RMSE,test: %6.2f\n",iter_local*j,loss_train[j],loss_test[j]))
        }        
        x<-seq(1,gbm_steps)
        if (j==2) {
          plot(x,loss_train,type="p",col="red")         
        } else if (i>2) {
          flush.console()
          points(x,loss_train,type="p",col="red")
          points(x,loss_test,type="p",col="green")
        }        
      }  
    ############################################# 
    # XGBOOST                                      #
    #############################################
    } else if (modname=="xgboost") {
      X = as.matrix(Xtrain)
      if (lossfn=="rmse") {
      xgb_params <- list(objective = "reg:linear",
                    #eval_metric = "rmse",
                    max.depth=parameters$depth,
                    eta=parameters$shseq,
                    subsample=parameters$subsample,
                    verbose=0,
                    nthread = parameters$nthread)
      #str(xgb_params)  
      n.tree <- parameters$iter
      fit = xgboost(param=xgb_params, data =X, label= ytrain,nrounds=n.tree,missing = NA,verbose=xgb_params$verbose)

      #oinfo(Xtrain)
      #oinfo(ytrain)
      #fit = xgboost(param=xgb_params, data =X, label= ytrain,nrounds=n.tree,missing = NA)
      
      }
      
    } else if (modname=="mars") {
      cat("TRAIN MARS: ") 
      if (lossfn=="rmse") {
        fit<-earth(x=Xtrain,y=ytrain,degree=2, nprune = 6, trace=0)
      } else {
        cat("mars: 0-1 distribution...")
        fit<-earth(x=Xtrain,y=ytrain,degree=2, nprune = 4, glm=list(family=binomial), trace=0)
      }
    ############################################# 
    # LINEAR                                    #
    #############################################
    } else {   
      if (lossfn=="rmse") {
        fit<-linRegTrain(Xtrain,ytrain,NULL,F)
      } else {
        fit <- glm(target ~ ., data=data.frame(Xtrain,target=ytrain),family=binomial(link="logit"))
        #fit<-glm.fit(Xtrain,ytrain,family=binomial(link="logit"))     
      }
    }
    ############################################# 
    # VALIDATION PART                          #
    #############################################
    if (modname=="rf") {
      if (lossfn=="rmse") {
        pred<-predict(fit,Xtest)
      } else {
        pred<-predict(fit,Xtest,type="vote")[,2]
        print(summary(pred))
      }    
    }
    else if (modname=="gam") {
      pred<-predict(fit,Xtest)
    }
    else if (modname=="svm") {
      cat("Predicting via SVM\n")
      pred<-predict(fit,Xtest)
    } else if (modname=="boost") {
      cat("TEST LINMODEL+RF: ")
      linpred<-linRegPredict(fit,Xtest,ytest,NULL)
      residues<-predict(rfmodel,Xtest)       
      pred<-linpred+residues
    } else if (modname=="gbm") {
      cat("TEST GBM with steps: ",gbm_steps,"\n")
      pred<-predict(fits[[gbm_steps]],Xtest,n.trees=parameters$iter,type="response")
      print(summary(data.frame(pred)))
    } else if (modname=="xgboost") {
      cat("TEST XGBOOST: ")
      if (lossfn=="rmse") {
        pred<-predict(fit,as.matrix(Xtest),outputmargin=T,predleaf=F, missing = NA)#XGBOOST
      }
    } else if (modname=="mars") {
      cat("TEST MARS: ")
      pred<-predict(fit, Xtest,type="response")
    } else {
      if (lossfn=="rmse") {
        pred<-linRegPredict(fit,Xtest,ytest,NULL)
      } else {
        pred<-predict(fit,newdata=Xtest,type=c("response"))
      }  
      
    }
    cat("n(test set):",length(ytest),"\n")
    if (lossfn=="rmse") {
      lossi<-compRMSE(pred,ytest)
    } else {
      lossi<-computeAUC(pred,ytest,F)
    }
    cat("Fold ",i," - Test Set Loss:",lossi,"\n")
    loss[i]<-lossi
  }
  if (lossfn=="rmse") {
    cat("Final RMSE:",mean(loss)," stdev: ",sd(loss),"\n")
  } else {
    cat("Final AUC:",mean(loss)," stdev: ",sd(loss),"\n")
  }
  return(mean(loss))
}

run_xgboost<-function(Xtrain,ytrain,analysis=F) {
  #current optimized parameters: ntree=2000, max.depth=10, eta=0.005, subsample=0.5 nsamples=12491 nvariables=30 RMSE(CV)=33.
  require(xgboost)
  require(methods)
  # Set necessary parameter
#   xgb_par_mp <- list(objective = "reg:linear",
#                   eval_metric = "rmse",
#                   max.depth=10,
#                   eta=0.005,
#                   silent=1, 
#                   subsample=0.5,
#                   nthread = 4)
  
  xgb_par <- list(objective = "reg:linear",
                     #eval_metric = "rmse",
                     max.depth=10,
                     eta=0.005,
                     silent=1, 
                     subsample=0.5,
                     nthread = 4,
                     verbose = 0)
  #gbm_parameters<-list(depth=c(10),shseq=c(0.005),iterations=c(200),minobsinnode=c(5,1))
  #gbm_grid(Xtrain,ytrain,repeatcv=2,lossfn="rmse",treemethod='xgboost',parameters=gbm_parameters) 
  n.tree<-2000
  #xval_oob(Xtrain,ytrain,Xtest=Xtrain,iterations=n.tree,nrfolds=8,repeatcv=2,lossfn="rmse",method="xgboost",intdepth=xgb_par$max.depth,sh=xgb_par$eta,oobfile='oob_res.csv',folds_from_file=NULL)
  xvalid(X,ytrain,nrfolds=8,modname="xgboost",loss="rmse",parameters=list(iter=n.tree,depth=xgb_par$max.depth,shseq=xgb_par$eta,subsample=xgb_par$subsample,nthread=xgb_par$nthread))
  #bst.cv = xgb.cv(param=parameters, data =as.matrix(Xtrain), label= ytrain, nfold = 8, nrounds=n.tree)
  #print(summary(bst.cv))
  model = xgboost(param=xgb_par, data =as.matrix(Xtrain), label= ytrain,nrounds=n.tree,missing = NA)
  #xgb.dump(model, 'xgb.model.dump', with.stats = FALSE)

  #trees<-xgb.model.dt.tree(feature_names = colnames(Xtrain),model = model,n_first_tree=n.tree)
  #saveXGB(model,"Tm[K]",colnames(Xtrain),filename="T(melting).propx")
  saveXGB(model,"density[g/cm3]",colnames(Xtrain),filename="density(crystal).propx")
  #test_data(model,Xtrain)
  if (analysis) {
    print(summary(trees))
    print(trees)
    
    diagram_info<-xgb.plot.tree(feature_names = colnames(Xtrain),  model = model,
                                n_first_tree = 1, width= 100, height=100)
    
    require(DiagrammeR)
    print(DiagrammeR(diagram_info$x$diagram))
  }
  
  
}


trainDBN<-function(lX,ly,lXtest=NULL,lytest=NULL){
  library(h2o)
  #setup heo library : http://cran.r-project.org/web/packages/h2o/h2o.pdf
  localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, Xmx = '2g')
  #localH2O <- h2o.init()
  
  
  data<-cbind(lX,target=ly)
  print(summary(data))
  data_h2o <- as.h2o(localH2O, data, key = 'data')
  
  model <- 
    h2o.deeplearning(x = colnames(lX),  # columns predictors
                     y = c("target"),   # columns label
                     data = data_h2o, # data in H2O format
                     #nfolds = 3, not implemented yet
                     classification = FALSE,
                     #activation = "MaxoutWithDropout", # or 'Tanh' TanhWithDropout
                     activation = "RectifierWithDropout", # or 'MaxoutWithDropout'
                     hidden = c(100,100), # three layers of 50 nodes
                     input_dropout_ratio = 0 ,# % of inputs dropout
                     hidden_dropout_ratios = c(0.0,0.0), # % for nodes dropout
                     #balance_classes = TRUE, 
                     seed=42,
                     #loss='MeanSquare',
                     l1 = c(0,1e-5), 
                     l2 = c(0,1e-5), 
                     rho = 0.99, 
                     epsilon = 1e-8,
                     train_samples_per_iteration = -2,
                     epochs = 100) # max. no. of epochs
  
  if (is.null(lXtest) && is.null(lytest)) {
    cat("Prediction on training data:\n")
    Xtest_h2o <- as.h2o(localH2O, X, key = 'train')
  } else {   
    cat("Test set provided:\n")
    Xtest<-matchByColumns(X,lXtest)
    Xtest_h2o <- as.h2o(localH2O, Xtest, key = 'test')
    ly<-ly_test
  }
    
  ## Collect cross-validation error
  MSE <- model@sumtable[[1]]$prediction_error   #If cvmodel is a grid search model
  #MSE <- model@model$valid_sqr_error            #If cvmodel is not a grid search model
  RMSE <- sqrt(MSE)
  CMRMSE <- CMRMSE + RMSE #column-mean-RMSE
  MSEs[resp] <- MSE
  RMSEs[resp] <- RMSE
  cat("\nCross-validated MSEs so far:", MSEs)
  cat("\nCross-validated RMSEs so far:", RMSEs)
  cat("\nCross-validated CMRMSE so far:", CMRMSE/resp)
  
  pred_h2o <- h2o.predict(model, Xtest_h2o)
  finalpred <- as.data.frame(pred_h2o)[,1]

  plot(finalpred,ly,col="blue",xlab = "predicted", ylab = "exp")
  abline(0,1, col = "black")
  rmse<-compRMSE(finalpred,y)
  cat("RMSE (DBN):",rmse,"\n")
  return(rmse)
  
}



printErrors<-function(model,lX,ly,id) {
  #correlationEllipses(cor(Xrf))
  se<-(model$predicted-model$y)^2
  er<-(model$predicted-model$y)
  pred<-data.frame(id,predicted=model$predicted,exp=ly,se,er)
  pred<-pred[with(pred, order(-se,decreasing=F)), ]
  write.table(pred,file="prediction_rf.csv",sep=";",row.names=FALSE)  
  aard<-compAARD(model$predicted,model$y)
  cat("AARD:",aard*100,"\n")
}

###PLOTTING###
makeBubblePlot<-function(model,lX,ly) {
  library(ggplot2)
  require(XML)
  if (!is.null(model$predicted)) {
    pred<-model$predicted#RF 
  } else if (!is.null(model$cv.fitted)) {
    pred<-model$cv.fitted#GBM CV    
  } else if (!is.null(model$fit)) {
    pred<-model$fit#GBM.FIT
  } else {
    pred<-model$fitted.values#LinReg  
  }
  cat("RMSE:",compRMSE(ly,pred))
  mdf<-data.frame(lX,target=ly,pred=pred)  
  b<-seq(100, 600, by = 100)
  p<-ggplot(data=mdf)+
    #geom_point(aes(x=pred,y=target,colour=M2,size=molweight))+
    #geom_point(aes(x=pred,y=target,colour=M2,size=molweight), alpha = 0.5, position = "jitter")+
    geom_point(aes(x=pred,y=target,colour=M2,size=ringbonds), alpha = 0.5)+
    #geom_point(aes(x=pred,y=target,size=tmult,colour=M2), alpha = 0.5)+
    #scale_colour_gradientn(colours=c("black","white"))+
    scale_colour_gradientn(colours=c("#00A352","red"))+
    scale_x_continuous("Tm, pred. [K]",breaks = b) +
    scale_y_continuous("Tm, exp. [K]",breaks = b)+
    #scale_size(guide="none",range=c(5,20))+
    scale_size(range=c(3,15))+
    #scale_colour_gradientn(colours=rainbow(2))+
    geom_abline(intercept=0, slope=1,size=1)+
    coord_fixed()+
    #xlim(100, 700)+
    #ylim(100, 700)+
    #scale_colour_gradientn(colours=c("black", "white"))+
    #theme(axis.text.x = element_text(colour="black",size=20,angle=90,hjust=.5,vjust=.5,face="plain"),
    #      axis.text.y = element_text(colour="black",size=20,angle=0,hjust=1,vjust=0,face="plain"))+
    theme_classic(base_size=20)
  print(p)  
  ggsave('mp2.png',  p)
  ggsave('mp1.png',  p, width = 300, height = 200, units = "mm",dpi = 300, scale = 1)
}

#http://is-r.blogspot.de/2012/11/plotting-correlation-ellipses.html
correlationEllipses<-function(cor){
  require(ellipse)  
  #ord <- order(cor[1, ])
  #cat(ord)
  #xc <- cor[ord, ord]
  xc <- cor
  colors <- c("#A50F15","#DE2D26","#FB6A4A","#FCAE91","#FEE5D9","white",
              "#EFF3FF","#BDD7E7","#6BAED6","#3182BD","#08519C")
  #colors <- c("#000000","#636363","#8A8A8A","#B0B0B0","#C9C9C9","white","#C9C9C9","#B0B0B0","#8A8A8A","#636363","000000")
  tmp<-colors[5*xc + 6]
  #   png(
  #     "corr.png",
  #     width     = 3.25,
  #     height    = 3.25,
  #     units     = "in",
  #     res       = 1200,
  #     pointsize = 4
  #   )
  #   par(
  #     mar      = c(5, 5, 2, 2),
  #     xaxs     = "i",
  #     yaxs     = "i",
  #     cex.axis = 2,
  #     cex.lab  = 1
  #   )
  plotcorr(xc,col=tmp)
  #  dev.off()
  #print(xc) 
}


plotTSNE<-function(X=NULL,labels=NULL,colors_col=NULL,nfeatures=20,nbins=2) {
  require(tsne)
  if (is.null(labels)) {
    X$levels = cut(X$rotatable_bonds,breaks=nbins)
    X$levels = cut(X$h_int,breaks=nbins)
  } else {
    X$levels = labels
  }
  print(summary(X))
  #oinfo(X$levels)
  #cat(X$levels)
  colors = rainbow(length(unique(colors_col)))
  names(colors) = unique(colors_col)
  ecb = function(x,y){ 
    plot(x,t='n'); 
    text(x,labels=X$levels, col=colors[colors_col])
    legend("topright", inset=c(-0.1,0), legend=unique(colors_col),pch=1, col=colors)}
  tsne_data = tsne(X[,1:nfeatures], epoch_callback = ecb, perplexity=20,epoch=20)
  # compare to PCA
  #dev.new()
  pca_data = princomp(X[,1:nfeatures])$scores[,1:2]
  plot(pca_data, t='n')
  text(pca_data, labels=X$levels,col=colors[colors_col])
  legend("topright", inset=c(-0.5,0), legend=unique(colors_col),pch=1, col=colors)
}

plotPartialdependence<-function(rf,ldata,n=3) {
  #PARTIAL DEPENDENCE
  imp<-importance(rf)
  print(summary(imp))
  impvar<-rownames(imp)[order(imp[, 1], decreasing=TRUE)]
  cat("impvar:",impvar,"\n")
  #op <- par(mfrow=c(2, n/2+1))
  #for (i in seq_along(impvar)) {
  for (i in 1:n) {
    cat("impvar:",impvar,"\n")  
    davar<-impvar[i]
    cat("impvar:",davar,"\n")
    partialPlot(rf,ldata,massprotbond)
#     partialPlot(ozone.rf, airquality, impvar[i], xlab=impvar[i],
#                 main=paste("Partial Dependence on", impvar[i]),
#                 ylim=c(30, 70))
    
  }
  #par(op)
}



#using sparse GF matrix
errorModel<-function() {
  require(glmnet)
  require(Matrix)
  
  edata = read.csv(file="error_fg.csv",sep=";")
  lX<-edata[,1:ncol(edata)-1]
  
  lX<-lX[,-c(1,2,3,4,5)]
  #cat(cs)
  cat("ncol before:",ncol(lX),"\n")
  lX<-lX[,which(colSums(lX)!=0)]
  lX<-removeColVar(lX,0.95)
  cat("ncol after:",ncol(lX),"\n")
  print(summary(lX))
  
  ly<-edata[,ncol(edata)]
  #selcol<-c(phosphonic_acid_derivative,conjugated_double_bond,heterocyclic,carboxylic_anhydride,carboxylic_acid,nitroso,S.Se,ketone,hbond_acceptors_CKD,cyanhydrine,aldehyde,tertiary_arom_amine)
  cvglm=cv.glmnet(as.matrix(lX),ly,family="gaussian",alpha=1.0,standardize = F,type.measure = "mse",nfolds = 10,intercept = T) 
  plot(cvglm)
  cat("RMSE best:",min(sqrt(cvglm$cvm)),"\n")
  coef.fit<-coef(cvglm,s=cvglm$lambda.min)[-1]
  threshhold=7
  index <- which(coef.fit >threshhold | coef.fit< -threshhold)
  good.coef <- coef.fit[index]
  cat(colnames(lX[,index]),"\n")
  #cat("Good ones",good.coef,"\n")
  cat("n coefficients with |c|> ",threshhold,":",length(good.coef),"\n") 
  #print((cvglm))
  #lX<-variableSelection(lX,ly,"forward",10)  
  
  #fit <- lm(ly~.,data.frame(lX,ly))
  #print(summary(fit)) # show results
  #layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
  #plot(fit)
  #print(coefficients(fit))  
}

