#!/usr/bin/Rscript
# http://stackoverflow.com/questions/12830160/how-do-i-change-the-colour-of-an-outline-in-a-ggplot-bubble-plot
# http://www.r-bloggers.com/top-250-movies-at-imdb/
setwd("D:/data_mining/")
library(ggplot2)
require(XML)

#url<-"http://www.ittfranking.com/gen/world/worldM_en.htm"
#mdf <- readHTMLTable(url, which = 2, stringsAsFactors = FALSE)
#print(head(mdf))


url<-"http://www.imdb.com/chart/top"
mdf <- readHTMLTable(url, which = 2, stringsAsFactors = FALSE)
pattern = "(.*) \\((.*)\\)$"
mdf = transform(mdf,Rating = as.numeric(Rating),Year= as.integer(substr(gsub(pattern, "\\2", Title), 1, 4)),Title  = gsub(pattern, "\\1", Title),Votes  = as.integer(gsub(",", "", Votes)))
mdf.lab<-mdf[which(mdf$Rating>8.5),]
mdf<-mdf[which(mdf$Rating<=8.5),]
#print(summary(mdf.lab))
#best.movies = best.movies[, c(4, 2, 3, 1)]
mdf<-mdf[order(-mdf$Votes), ]
print(head(mdf,50))
print(summary(mdf))
cat("cor:",cor(mdf$Rating,mdf$Votes))
p<-ggplot(mdf, aes(x = Year, y = Rating))+geom_text(data = mdf.lab, aes(x=Year,y=Rating, label = Title),size=3)+geom_point(aes(size = Votes), alpha = 0.5, position = "jitter", color = "darkgreen")+geom_point(data = mdf.lab,aes(size = Votes), alpha = 0.5, color = "red", position = "jitter")+scale_size(range = c(3, 15))+theme_classic()
print(p)