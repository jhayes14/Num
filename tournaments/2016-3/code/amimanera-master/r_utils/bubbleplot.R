#!/usr/bin/Rscript
# http://stackoverflow.com/questions/12830160/how-do-i-change-the-colour-of-an-outline-in-a-ggplot-bubble-plot
# http://www.r-bloggers.com/top-250-movies-at-imdb/
setwd("D:/data_mining/")
library(ggplot2)
require(XML)

bubbleplot<-function(mdf,xaxis,yaxis,bubble,mdf.lab=NULL,mdflabel) {
  if (!is.null(mdf.lab)) {
    p<-ggplot(mdf, aes(x = xaxis, y = yaxis))+geom_text(data = mdf.lab, aes(x=xaxis,y=yaxis, label = mdflabel),size=3)+geom_point(aes(size = bubble), alpha = 0.5, position = "jitter", color = "darkgreen")+geom_point(data = mdf.lab,aes(size = bubble), alpha = 0.5, color = "red", position = "jitter")+scale_size(range = c(3, 15))+theme_classic()
  } else {
    p<-ggplot(mdf, aes(x = xaxis, y = yaxis))+geom_point(aes(size = bubble), alpha = 0.5, position = "jitter", color = "darkgreen")+scale_size(range = c(3, 15))+theme_classic()
  }
  print(p)
}