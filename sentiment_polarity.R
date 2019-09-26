sentiment <- read.csv("sentiment.csv")

str(sentiment)
table(sentiment$polarity)
sentiment$sentences <- as.character(sentiment$sentences)
sentiment$polarity <- factor(sentiment$polarity,
                             levels=c(0,1),
                             labels=c("Negative","Positive"))

str(sentiment)   

#shuffle the sentences 
set.seed(12345)
sentiment <- sentiment[order(runif(2748)) ,]

#create corpus and clean corpus
install.packages("tm")
install.packages("SnowballC")
library(tm)
library(SnowballC)

text_corpus <- Corpus(VectorSource(sentiment$sentences))
print(text_corpus)
inspect(text_corpus[1:6])

text_corpus_clean <- tm_map(text_corpus, tolower)
text_corpus_clean <- tm_map(text_corpus_clean, removeNumbers)
text_corpus_clean <- tm_map(text_corpus_clean,removePunctuation)
text_corpus_clean <- tm_map(text_corpus_clean, removeWords,stopwords())
# text_corpus_clean <- tm_map(text_corpus_clean,stemDocument)
text_corpus_clean <- tm_map(text_corpus_clean,stripWhitespace)

inspect(text_corpus[1:6])
inspect(text_corpus_clean[1:6])

#create sparse matrix
text_dtm <- DocumentTermMatrix(text_corpus_clean)        
dim(text_dtm)
text_dtm
inspect(text_dtm[0:10, 1:15])

#sort the words based on frequency
freq <- sort(colSums(as.matrix(text_dtm)),decreasing = TRUE)
library(ggplot2)
wf <- data.frame(words=names(freq),freq=freq)
head(wf)

#showing wordcloud of the positive and negative sentiments
library(wordcloud)
install.packages("RColorBrewer")
library(RColorBrewer)
par("mar")
par(mar=c(1,1,1,1))

positive <- subset(sentiment,polarity =="Positive")
negative <- subset(sentiment,polarity =="Negative")
str(positive)

wordcloud(positive$sentences, max.words = 100,random.order = FALSE, scale=c(3,0.5),colors = brewer.pal(7,"Dark2"))
wordcloud(negative$sentences,max.words = 100, random.order = FALSE,scale=c(3,0.5),colors = brewer.pal(7,"Dark2"))

#convert word count to factor 
convert <- function(x) {
  x<- ifelse(x>1,1,0)
  x<- factor(x, levels = c(0,1),
             labels = c("No","Yes"))
  return(x)
}

#convert the sparse matrix into data.frame
text_new <- as.data.frame(as.matrix(apply(text_dtm,2,convert)))


#merge polarity into the text_new data.frame as class
text_new$Class = sentiment$polarity
str(text_new$Class)

head(text_new)
dim(text_new)  #Class(polarity) is in the 5110th column of the matrix
head(text_new[[5110]])

#create traning and test data
set.seed(222)
split <- sample(2,nrow(sentiment),replace = TRUE, prob = c(0.7,0.3))
text_train <- text_new[split==1,]
text_test <- text_new[split==2,]

prop.table(table(text_train$Class))
prop.table(table(text_test$Class))

# building Naive Bayes model
library(e1071)
NBclassifier <- naiveBayes(text_train, text_train$Class)
NBclassifier

#checking performance
NB_pred <- predict(NBclassifier,text_test)
library(caret)
confusionMatrix(NB_pred,text_test$Class)
library(gmodels)
CrossTable(text_test$Class,NB_pred,
           prop.chisq = FALSE,
           prop.t=FALSE,
           dnn=c("actual","predicted"))

#imporving the perfomance by adding laplace=1
NBclassifier2 <- naiveBayes(text_train, text_train$Class,laplace = 1)
#performance
NB_pred2 <- predict(NBclassifier2,text_test)

confusionMatrix(NB_pred2,text_test$Class)

CrossTable(text_test$Class,NB_pred,
           prop.chisq = FALSE,
           prop.t=FALSE,
           dnn=c("actual","predicted"))


#try to build the model using random forest
library(randomForest)
rf_classifier = randomForest(x = text_train[-5110],
                             y = text_train$Class,
                             ntree = 300)
rf_classifier

rf_pred = predict(rf_classifier, newdata = text_test[-5110])

# Making the Confusion Matrix
confusionMatrix(table(rf_pred,text_test$Class))
CrossTable(text_test$Class,rf_pred,
           prop.chisq = FALSE,
           prop.t=FALSE,
           dnn=c("actual","predicted"))

