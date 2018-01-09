# five different resampling methods that you can use to evaluate the accuracy of your data in R.

# *************Data splitting*************
# -> involves partitioning the data into an explicit training dataset used to prepare
#    the model and an unseen test dataset used to evaluate the models performance on unseen data.
# -> Useful when you have a very large dataset so that the test dataset can provide a meaningful estimation of
#    performance, or for when you are using slow methods and need a quick approximation of performance.

# Splits the iris dataset so that 80% is used for training a Naive Bayes model and 
# 20% is used to evaluate the models performance.
library(caret)
library(klaR)
data(iris)
# define an 80%/20% train/test split of the dataset
trainIndex <- createDataPartition(iris$Species, p=0.80, list=FALSE)
dataTrain <- iris[ trainIndex,]
dataTest <- iris[-trainIndex,]
# train a naive Bayes model
fit <- NaiveBayes(Species~., data=dataTrain)
# make predictions
predictions <- predict(fit, dataTest[,1:4])
# summarize results
confusionMatrix(predictions$class, dataTest$Species)

# *************Bootstrap resampling*************
# -> involves taking random samples from the dataset (with replacement) against which to evaluate the model.
# -> In aggregate, the results provide an indication of the variance of the models performance.
# -> Will create one/> new training datasets that'll also contain n egs, some of which are repeated. 
# -> One advantage of bootstrap over cross-validation is that it tends to work better with very small datasets. 
# -> Additionally, bootstrap sampling has applications beyond performance measurement. 

library(caret)
data(iris)
# define training control
trainControl <- trainControl(method="boot", number=100)
# evalaute the model
fit <- train(Species~., data=iris, trControl=trainControl, method="nb")
# display the results
print(fit)

# *************k-fold cross validation*************
# -> involves splitting the dataset into k-subsets
# -> Randomly divides the data into k completely separate random partitions called folds.
# -> the most common convention is to use 10-fold cross-validation
# -> robust method for estimating accuracy, and the size of k can tune the amount of bias in
#    the estimate, with popular values set to 5 and 10. 

# you will see the estimated of the accuracy of the model using 10-fold
cross validation.
library(caret)
data(iris)
# define training control
trainControl <- trainControl(method="cv", number=10)
# evaluate the model
fit <- train(Species~., data=iris, trControl=trainControl, method="nb")
# display the results
print(fit)

# *************Repeated k-fold Cross Validation*************
# -> The process of splitting the data into k-folds can be repeated a number of times
# ->  repeatedly applying k-fold CV and averaging the results.

library(caret)
data(iris)
# define training control
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
# evaluate the model
fit <- train(Species~., data=iris, trControl=trainControl, method="nb")
# display the results
print(fit)

# *************Leave One Out Cross Validation (LOOCV)*************
# -> A data instance is left out and a model constructed on all other data instances in the training set. 
# -> This is repeated for all data instances.

library(caret)
data(iris)
# define training control
trainControl <- trainControl(method="LOOCV")
# evaluate the model
fit <- train(Species~., data=iris, trControl=trainControl, method="nb")
# display the results
print(fit)