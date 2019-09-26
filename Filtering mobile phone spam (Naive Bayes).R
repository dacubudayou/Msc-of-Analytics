sms_raw <- read.csv("sms_spam.csv",stringsAsFactors = FALSE)
str(sms_raw)


sms_raw$type <- factor(sms_raw$type)
str(sms_raw$type)
table(sms_raw$type)

install.packages("tm")
library(tm)

#we only use corpus for sms_raw$text, not for the sms_raw$type!
sms_corpus <- Corpus(VectorSource(sms_raw$text))
print(sms_corpus)
inspect(sms_corpus[1:3])
corpus_clean <- tm_map(sms_corpus,tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean,removeWords,stopwords())
corpus_clean <- tm_map(corpus_clean,removePunctuation)
corpus_clean <- tm_map(corpus_clean,stripWhitespace)

#to see the cleaning result by comparing the first 3 text
inspect(sms_corpus[1:3])
inspect(corpus_clean[1:3])

sms_dtm <- DocumentTermMatrix(corpus_clean)
inspect(sms_dtm[0:10, 1:15])

#creating training and test data

#for raw data frame:
sms_raw_train <- sms_raw[1:4181,]
sms_raw_test <- sms_raw[4182:5574,]

#for document-term matrix:
sms_dtm_train <- sms_dtm[1:4181,]
sms_dtm_test <- sms_dtm[4182:5574,]

#for corpus:
sms_corpus_train <- corpus_clean[1:4181]
sms_corpus_test <- corpus_clean[4182:5574]

prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))

#visualising text data using wordcloud
install.packages("wordcloud")
library(wordcloud)

wordcloud(sms_corpus_train, min.freq = 40, radom.order = FALSE)


spam <- subset(sms_raw_train, type == "spam")
ham <- subset(sms_raw_train,type == "ham")

wordcloud(spam$text, max.words = 40,scale =c(3,0.5),random.order = FALSE)
wordcloud(ham$text, max.words = 40, scale = c(3,0.5),random.order = FALSE)

#data preparation - creating indicator features for frequent words
install.packages("hash")
library(hash)
findFreqTerms(sms_dtm_train,5)
sms_dict <- (findFreqTerms(sms_dtm_train,5))

sms_train <- DocumentTermMatrix(sms_corpus_train,list(dictionary = sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test,list(dictionary = sms_dict))

convert_counts <- function(x){
  x <- ifelse(x>0, 1,0)
  x <- factor(x,levels =c(0,1),labels = c("No","Yes"))
  return(x)
}

sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_test, MARGIN = 2, convert_counts)


#training a model using naive Bayes algorithem
install.packages("e1071")
library(e1071)
sms_classifier <- naiveBayes(sms_train,sms_raw_train$type)

#evaluting the model performance
sms_test_pred <- predict(sms_classifier,sms_test)
library(gmodels)
CrossTable(sms_test_pred,sms_raw_test$type, prop.chisq = FALSE,
  prop.t = FALSE, dnn = c("predicted","actual") )
library(caret)
confusionMatrix(sms_test_pred,sms_raw_test$type)


#improving the model perfomance
sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$type, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2,sms_raw_test$type,prop.chisq = FALSE,
           prop.t = FALSE, dnn= c("predicted","actual"))
confusionMatrix(sms_test_pred2,sms_raw_test$type)
