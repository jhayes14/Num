#!/usr/bin/Rscript
options(scipen = 10)

###GENERAL SETTINGS###
set.seed(41)

if (.Platform$OS.type=='unix') {
  setwd(".")
  source("./qspr.R")
} 

#https://www.kaggle.com/ademyttenaere/caterpillar-tube-pricing/build-complete-train-and-test-db
merge_only<-function(useCompfiles=FALSE) {
  require(caret)
  require(randomForest)
  base = "../data/"
  test = read.csv(paste0(base, "test.csv"))
  train = read.csv(paste0(base, "train.csv"))

  train$id = -(1:nrow(train))
  test$cost = 0

  dFull = rbind(train, test)

  dFull = merge(dFull, read.csv(paste0(base, "bill_of_materials.csv"), quote = ""), by = "tube_assembly_id", all.x = TRUE)
  dFull = merge(dFull, read.csv(paste0(base, "specs.csv"), quote = ""), by = "tube_assembly_id", all.x = TRUE)
  dFull = merge(dFull, read.csv(paste0(base, "tube.csv"), quote = ""), by = "tube_assembly_id", all.x = TRUE)
  compFiles = dir(base)[grep("comp_", dir(base))]

  if (useCompfiles==TRUE) {
    idComp = 1
    keyMerge = 0
    for(idComp in 1:8){
      for(f in compFiles){
	d = read.csv(paste0(base, f), sep = ',', quote = "")
	names(d) = paste0(names(d), "_", keyMerge)
	dFull = merge(dFull, d, by.x = paste0("component_id_", idComp), by.y = paste0("component_id_", keyMerge), all.x = TRUE)
	keyMerge = keyMerge + 1
      }
      cat("idComp = ", idComp, " - nrow(dFull) = ", nrow(dFull), " and ncol(dFull) = ", ncol(dFull), "\n")
    }
  }
  
  ### Clean NA values
    for(i in 1:ncol(dFull)){
      if(is.numeric(dFull[,i])){
	dFull[is.na(dFull[,i]),i] = -1
      }else{
	dFull[,i] = as.character(dFull[,i])
	dFull[is.na(dFull[,i]),i] = "NAvalue"
	dFull[,i] = as.factor(dFull[,i])
      }
    }

    ### Clean variables with too many categories
    threshold<-30
    for(i in 1:ncol(dFull)){
      if(!is.numeric(dFull[,i])){
	freq = data.frame(table(dFull[,i]))
	freq = freq[order(freq$Freq, decreasing = TRUE),]
	dFull[,i] = as.character(match(dFull[,i], freq$Var1[1:threshold]))
	dFull[is.na(dFull[,i]),i] = "rareValue"
	dFull[,i] = as.factor(dFull[,i])
      }
    }
  
  #oinfo(dFull)
  #dFull<-removeColVar(dFull,0.95)
  oinfo(dFull)
  
  #remove zerovars via caret
  tmp<-nearZeroVar(dFull)
  if (length(tmp) > 0) {
    dFull <- dFull[, -tmp] 
  }
  oinfo(dFull)
  
  test = dFull[which(dFull$id > 0),]
  train = dFull[which(dFull$id < 0),]

  #test = test[,-match("id", names(test))]
  #train = train[,-match("id", names(train))]
  
  y = log(train$cost + 1.0)
  train<-train[,-match(c("id"), names(train))]
  test<-test[,-match(c("id"), names(test))]
  
  cat("Final train dataset : ", nrow(train), " rows and ", ncol(train), " columns\n")
  cat("Final test dataset : ", nrow(test), " rows and ", ncol(test), " columns\n")

  oinfo(train)
  oinfo(test)
  oinfo(y)
  print(summary(train))
  #print(summary(test))
  cat(colnames(train))
  #train<-rf_select(train[,-train$cost],train$cost,250,65)
  #test<-matchByColumns(train,test)
  
  write.table(test, "../data/test_R.csv", sep = ";", row.names = FALSE, quote = FALSE)
  write.table(train, "../data/train_R.csv", sep = ";", row.names = FALSE, quote = FALSE)
  train<-train[,-match(c("cost"), names(train))]
  test<-test[,-match(c("cost"), names(test))]
  
  rf1 <- randomForest(train,y,ntree=150,importance = T,do.trace = 2)
  imp<-importance(rf1, type=1)
  increaseMSE<-data.frame(imp)  
  index<-order(-increaseMSE[,1])
  dd<-increaseMSE[index,,drop=F]

  print(dd)
  str(dd)
  
  varImpPlot(rf1,n.var=65,type=1,main="")
  print(rf1)
  
}


#https://www.kaggle.com/ademyttenaere/caterpillar-tube-pricing/0-2748-with-rf-and-log-transformation
merge_run<-function() {
    ###
    ### Build train and test db
    ###

    ### Load train and test
    test = read.csv("../data/test.csv")
    train = read.csv("../data/train.csv")

    train$id = -(1:nrow(train))
    test$cost = 0

    train = rbind(train, test)

    ### Merge datasets if only 1 variable in common
    continueLoop = TRUE
    while(continueLoop){
      continueLoop = FALSE
      for(f in dir("../data/")){
	d = read.csv(paste0("../data/", f))
	commonVariables = intersect(names(train), names(d))
	if(length(commonVariables) == 1){
	  train = merge(train, d, by = commonVariables, all.x = TRUE)
	  continueLoop = TRUE
	  print(dim(train))
	}
      }
    }

    ### Clean NA values
    for(i in 1:ncol(train)){
      if(is.numeric(train[,i])){
	train[is.na(train[,i]),i] = -1
      }else{
	train[,i] = as.character(train[,i])
	train[is.na(train[,i]),i] = "NAvalue"
	train[,i] = as.factor(train[,i])
      }
    }

    ### Clean variables with too many categories
    threshold<-15
    for(i in 1:ncol(train)){
      if(!is.numeric(train[,i])){
	freq = data.frame(table(train[,i]))
	freq = freq[order(freq$Freq, decreasing = TRUE),]
	train[,i] = as.character(match(train[,i], freq$Var1[1:threshold]))
	train[is.na(train[,i]),i] = "rareValue"
	train[,i] = as.factor(train[,i])
      }
    }

    test = train[which(train$id > 0),]
    train = train[which(train$id < 0),]

    oinfo(train)
    oinfo(test)

    #print(summary(train))
    #print(summary(test))
    
    write.csv(train,"../data/train_R2.csv",row.names=FALSE,quote=FALSE)
    write.csv(test,"../data/test_R2.csv",row.names=FALSE,quote=FALSE)

    ###
    ### Evaluate RF predictions by splitting the train db in 80%/20%
    ###

    ### Randomforest
    library(randomForest)

    # dtrain_cv = train[which(train$id %% 5 > 0),]
    # dtest_cv = train[which(train$id %% 5 == 0),]
    # 
    # ### Train randomForest on dtrain_cv and evaluate predictions on dtest_cv
    # set.seed(123)
    # rf1 = randomForest(dtrain_cv$cost~., dtrain_cv[,-match(c("id", "cost"), names(dtrain_cv))], ntree = 10, do.trace = 2)
    # 
    # pred = predict(rf1, dtest_cv)
    # sqrt(mean((log(dtest_cv$cost + 1) - log(pred + 1))^2)) # 0.2589951
    # 
    # ### With log transformation trick
    # set.seed(123)
    # rf2 = randomForest(log(dtrain_cv$cost + 1)~., dtrain_cv[,-match(c("id", "cost"), names(dtrain_cv))], ntree = 10, do.trace = 2)
    # pred = exp(predict(rf2, dtest_cv)) - 1
    # 
    # sqrt(mean((log(dtest_cv$cost + 1) - log(pred + 1))^2)) # 0.2410004

    ### Train randomForest on the whole training set
    #y = log(train$cost + 1)
    #X = train[,-match(c("id", "cost"), names(train))]
    #print(summary(X))
    #rf = randomForest(X,y,ntree = 20, do.trace = 2)
    #rf = randomForest(log(train$cost + 1)~., train[,-match(c("id", "cost"), names(train))], ntree = 20, do.trace = 2)

    #pred = exp(predict(rf, test)) - 1.0

    #submitDb = data.frame(id = test$id, cost = pred)
    #submitDb = aggregate(data.frame(cost = submitDb$cost), by = list(id = submitDb$id), mean)

    #write.csv(submitDb, "submit.csv", row.names = FALSE, quote = FALSE)
}


xgb_boost<-function () {

    # This is implementation of XGboost model in R
    # library required

    library(data.table)
    library(xgboost)
    library(Matrix)
    library(methods)

    # you must know why I am using set.seed()
    set.seed(546)


    # Importing data into R

    train  <- read.csv("../data/train_set.csv",header = T)
    test  <- read.csv("../data/test_set.csv",header = T)
    bom  <- read.csv("../data/bill_of_materials.csv",header = T)
    specs  <- read.csv("../data/specs.csv",header = T)
    tube  <- read.csv("../data/tube.csv",header = T)

    # Merging the data

    train$id  <- -(1:nrow(train))
    test$cost  <- 0

    data  <- rbind(train,test)

    data  <- merge(data,tube,by="tube_assembly_id",all = T)
    data  <- merge(data,bom,by="tube_assembly_id",all = T)
    data  <- merge(data,specs,by="tube_assembly_id",all = T)

    # extracting year and month for quote_date

    data$quote_date  <- strptime(data$quote_date,format = "%Y-%m-%d", tz="GMT")
    data$year <- year(as.IDate(data$quote_date))
    data$month <- month(as.IDate(data$quote_date))
    data$week <- week(as.IDate(data$quote_date))

    # dropping variables
    data$quote_date  <- NULL
    data$tube_assembly_id  <- NULL


    # converting NA in to '0' and '" "' for mode Matrix Generation

    for(i in 1:ncol(data)){
      if(is.numeric(data[,i])){
	data[is.na(data[,i]),i] = 0
      }else{
	data[,i] = as.character(data[,i])
	data[is.na(data[,i]),i] = " "
	data[,i] = as.factor(data[,i])
      }
    }


    # converting data.frame to sparse matrix for modelling

    train  <- data[which(data$id < 0), ]
    test  <- data[which(data$id > 0), ]

    ids  <- test$id
    cost  <- train$cost

    #dropping some more variables

    train$id  <- NULL 
    test$id  <- NULL
    #train$cost  <- 0
    test$cost  <- NULL

    print(summary(train))

    # this is a very crude way of generating sparse matrix and might take a bit time
    # if anybody has a better way feel free to comment;)

    tr.mf  <- model.frame(as.formula(paste("cost ~",paste(names(train),collapse = "+"))),train)
    tr.m  <- model.matrix(attr(tr.mf,"terms"),data = train)
    tr  <- Matrix(tr.m)
    t(tr)
    print(tr)
    cat(dim(tr))
    print(dim(tr))
  
    te.mf  <- model.frame(as.formula(paste("~",paste(names(test),collapse = "+"))),test)
    te.m  <- model.matrix(attr(te.mf,"terms"),data = test)
    te  <- Matrix(te.m)
    t(te)



    # generating xgboost model 

    # tr.x  <- xgb.DMatrix(tr,lable=log(names(train)+1))
    cost.log  <- log(cost+1) # treating cost as log transfromation is working good on this data set

    tr.x  <- xgb.DMatrix(tr,label = cost.log)
    te.x  <- xgb.DMatrix(te)


    # parameter selection
    par  <-  list(booster = "gblinear",
		  objective = "reg:linear",
		  min_child_weight = 6,
		  gamma = 2,
		  subsample = 0.85,
		  colsample_bytree = 0.75,
		  max_depth = 10,
		  verbose = 1,
		  scale_pos_weight = 1)


    #selecting number of Rounds
    n_rounds= 200


    #modeling

    x.mod.t  <- xgb.train(params = par, data = tr.x , nrounds = n_rounds)
    pred  <- predict(x.mod.t,te.x)
    head(pred)

    for(i in 1:50){
      x.mod.t  <- xgb.train(par,tr.x,n_rounds)
      pred  <- cbind(pred,predict(x.mod.t,te.x))
    }

    pred.sub  <- exp(rowMeans(pred))-1


    # generating data frame for submission

annual_usage
    sub.file = data.frame(id = ids, cost = pred.sub)
    sub.file = aggregate(data.frame(cost = sub.file$cost), by = list(id = sub.file$id), mean)

    write.csv(sub.file, "submit.csv", row.names = FALSE, quote = FALSE)
}


feature_importance<-function() {
    library(randomForest)
    Xtrain <- read.csv(file="../data/Xtrain.csv")
    #Xtrain <- subset(Xtrain,select=-tube_assembly_id)
    ytrain <- read.csv(file="../data/ytrain.csv")
    ytrain = log(ytrain$cost + 1)
    #ytrain<-as.factor(ytrain$class)

    print(summary(Xtrain))
    print(summary(ytrain))
    oinfo(ytrain)

    #Xtrain<-boruta_select(Xtrain,ytrain)
    Xtrain<-rf_select(Xtrain,ytrain,250,70)

    rf1 <- randomForest(Xtrain,ytrain,ntree=100,importance = T,do.trace = 2)
    imp<-importance(rf1, type=1)
    increaseMSE<-data.frame(imp)  
    index<-order(-increaseMSE[,1])
    dd<-increaseMSE[index,,drop=F]
    #filter variable which do no contribute 
    print(dd)
    str(dd)
    
    varImpPlot(rf1,n.var=30,type=1,main="")
    print(rf1)
}

train_oob<-function() {
    library(randomForest)
    Xtrain <- read.csv(file="../data/Xtrain.csv")
    Xtest <- read.csv(file="../data/Xtest.csv")
    ytrain <- read.csv(file="../data/ytrain.csv")
    ta<-read.csv(file="../data/ta.csv")
    ytrain <- log(ytrain$cost + 1)
    
    Xtrain <-removeZeroVars(Xtrain)
    Xtest <-removeZeroVars(Xtest)
    Xtrain <- matchByColumns(Xtest,Xtrain)
    oinfo(Xtrain)
    oinfo(Xtest)
    
    cat(colnames(Xtrain))
    cat(colnames(Xtest))
    
    #print(summary(Xtrain))
    #print(summary(Xtest))
    #Xvalidation
    #xval_oob
    #outlierDetection
    #makeBubblePlot
    compareViaPrinComp(Xtrain,Xtest)
    
    
}

#merge_only(useCompfiles=FALSE)
merge_run()
#feature_importance()
#train_oob()
#xgb_boost()
