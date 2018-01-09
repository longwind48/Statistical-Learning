##### Chapter 6: Regression Methods -------------------

# ************Linear Regression************

## Understanding regression ----
## Example: Space Shuttle Launch Data ----
launch <- read.csv("challenger.csv")

# confirming the regression line using the lm function (not in text)
model <- lm(distress_ct ~ temperature, data = launch)
model
summary(model)

# creating a simple multiple regression function
reg <- function(y, x) {
  x <- as.matrix(x)
  x <- cbind(Intercept = 1, x)
  b <- solve(t(x) %*% x) %*% t(x) %*% y
  colnames(b) <- "estimate"
  print(b)
}

# examine the launch data
str(launch)

# test regression model with simple linear regression
reg(y = launch$distress_ct, x = launch[2])

# use regression model with multiple regression
reg(y = launch$distress_ct, x = launch[2:4])

# confirming the multiple regression result using the lm function (not in text)
model <- lm(distress_ct ~ temperature + pressure + launch_id, data = launch)
model

## Example: Predicting Medical Expenses ----
## Step 2: Exploring and preparing the data ----
insurance <- read.csv("insurance.csv", stringsAsFactors = TRUE)
str(insurance)

# model's dependent variable is expenses, which measures the medical costs each person charged to the insurance plan for the year
# Prior to building a regression model, it is often helpful to check for normality, because model fits better with normality.

# summarize the charges variable
summary(insurance$expenses)
# mean value is greater than the median, implies that the distribution of insurance expenses is right-skewed.

# histogram of insurance charges
hist(insurance$expenses)
# shows a right-skewed distribution. majority have yearly medical expenses between zero and $15,000,

# Regression models require that every feature is numeric, yet we have three factor-type features in our dataframe. 
# table of region
table(insurance$region)
# yet we have three factor-type features in our data frame.

# it can be useful to determine how the independent variables are related to the dependent variable and each other. 
# exploring relationships among features: correlation matrix
cor(insurance[c("age", "bmi", "children", "expenses")])
# None of the correlations in the matrix are considered strong
# age and bmi appear to have a weak positive correlation, meaning that as someone ages, their body mass tends to increase.
# also a moderate positive correlation between age and expenses, bmi and expenses, and children and expenses. 
# These associations imply that as age, body mass, and number of children increase, the expected cost of insurance goes up.


# visualing relationships among features: scatterplot matrix
pairs(insurance[c("age", "bmi", "children", "expenses")])
# The relationship between age and expenses displays several relatively straight lines, 
# while the bmi versus expenses plot has two distinct groups of points. 
# It is difficult to detect trends in any of the other plots.

# more informative scatterplot matrix
library(psych)
pairs.panels(insurance[c("age", "bmi", "children", "expenses")])
# Above the diagonal, the scatterplots have been replaced with a correlation matrix.
# On the diagonal, a histogram depicting the distribution of values for each feature is shown. 
# The oval-shaped object on each scatterplot is a correlation ellipse, which provides a visualization of correlation strength. 
# the more it is stretched, the stronger the correlation.
# The curve drawn on the scatterplot is called a loess curve, indicates general relationship between the x and y axis variables.
# the loess curve for age and bmi is a line sloping gradually up, implying that body mass increases with age,
# curve for age and children is an upside-down U, oldest and youngest people in the sample have fewer children on the
# insurance plan than those around middle age.


## Step 3: Training a model on the data ----
ins_model <- lm(expenses ~ age + children + bmi + sex + smoker + region,
                data = insurance)
ins_model <- lm(expenses ~ ., data = insurance) # this is equivalent to above

# see the estimated beta coefficients
ins_model
# for each additional year of age, we would expect $256.80 higher medical expenses on average, holding everything else constant

## Step 4: Evaluating model performance ----
# see more detail about the estimated beta coefficients
summary(ins_model)
# -> Since a residual is equal to the true value minus the predicted value, the maximum error of
#    29981.7 suggests that the model under-predicted expenses by nearly $30,000 for at least one observation.
# -> Small p-values suggest that the true coefficient is very unlikely to be zero, which means that the feature is 
#    extremely unlikely to have no relationship with the dependent variable.
# -> Our model has several highly significant variables, and they seem to be related to the outcome in logical ways
# -> Multiple R-squared value (also called the coefficient of determination) provides a measure of how well our model 
#    as a whole explains the values of the dependent variable. 
# -> The model explains nearly 75 percent of the variation in the dependent variable.
# -> our model is performing fairly well. 


## Step 5: Improving model performance ----
# -> the relationship between an independent variable and the dependent variable is assumed to be linear, 
#    yet this may not necessarily be true.
# -> To account for a non-linear relationship,
# add a higher-order "age" term
insurance$age2 <- insurance$age^2

# Transformation - converting a numeric variable to a binary indicator
# -> create a binary obesity indicator variable that is 1 if the BMI is at least 30, and 0 if less. 
# add an indicator for BMI >= 30
insurance$bmi30 <- ifelse(insurance$bmi >= 30, 1, 0)

# For instance, smoking and obesity may have harmful effects separately, but it is reasonable to 
# assume that their combined effect may be worse than the sum of each one alone.
# -> When two features have a combined effect, this is known as an interaction.

# create final model
ins_model2 <- lm(expenses ~ age + age2 + children + bmi + sex +
                   bmi30*smoker + region, data = insurance)

summary(ins_model2)
# -> the R-squared value has improved from 0.75 to about 0.87
# -> The higher-order age2 term is statistically significant, as is the obesity indicator, bmi30. 
# -> The interaction between obesity and smoking suggests a massive effect
#   in addition to the increased costs of over $13,404 for smoking alone, obese smokers spend another $19,810 per year. 

# The lm() function is in the stats package and creates a linear regression model using ordinary least squares
library(mlbench)
data(BostonHousing)
# fit model
fit <- lm(medv~., BostonHousing)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, BostonHousing)
# summarize accuracy
mse <- mean((BostonHousing$medv - predictions)^2)
print(mse)

# Example of linear regression algorithm in caret.
library(caret)
library(mlbench)
data(BostonHousing)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
fit.lm <- train(medv~., data=BostonHousing, method="lm", metric="RMSE", preProc=c("center",
                                                                                  "scale"), trControl=trainControl)
# summarize fit
print(fit.lm)

# ************Logistic Regression************
# creates a generalized linear model for regression or classification. 
library(mlbench)
data(PimaIndiansDiabetes)
# fit model
fit <- glm(diabetes~., data=PimaIndiansDiabetes, family=binomial(link='logit'))
# summarize the fit
print(fit)
# make predictions
probabilities <- predict(fit, PimaIndiansDiabetes[,1:8], type='response')
predictions <- ifelse(probabilities > 0.5,'pos','neg')
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)

library(caret)
library(mlbench)
data(PimaIndiansDiabetes)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
fit.glm <- train(diabetes~., data=PimaIndiansDiabetes, method="gpls", metric="Accuracy",
                 preProc=c("center", "scale"), trControl=trainControl)
# summarize fit
print(fit.glm)

# ************Linear Discriminant Analysis************
library(MASS)
library(mlbench)
data(PimaIndiansDiabetes)
fit <- lda(diabetes~., data=PimaIndiansDiabetes)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, PimaIndiansDiabetes[,1:8])$class
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)

library(caret)
library(mlbench)
data(PimaIndiansDiabetes)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
fit.lda <- train(diabetes~., data=PimaIndiansDiabetes, method="lda", metric="Accuracy",
                 preProc=c("center", "scale"), trControl=trainControl)
# summarize fit
print(fit.lda)

# ************Regularized Regression************

# Example of GLMNET algorithm for regression
library(glmnet)
library(mlbench)
data(BostonHousing)
BostonHousing$chas <- as.numeric(as.character(BostonHousing$chas))
x <- as.matrix(BostonHousing[,1:13])
y <- as.matrix(BostonHousing[,14])
# fit model
fit <- glmnet(x, y, family="gaussian", alpha=0.5, lambda=0.001)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, x, type="link")
# summarize accuracy
mse <- mean((y - predictions)^2)
print(mse)

# Example of GLMNET algorithm for classification
library(glmnet)
library(mlbench)
data(PimaIndiansDiabetes)
x <- as.matrix(PimaIndiansDiabetes[,1:8])
y <- as.matrix(PimaIndiansDiabetes[,9])
# fit model
fit <- glmnet(x, y, family="binomial", alpha=0.5, lambda=0.001)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, x, type="class")
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)

# can also be configured to perform three important types of regularization: lasso, ridge and elastic net by configuring the alpha parameter to 1, 0 or in [0,1] respectively.
# glmnet implementation can be used in caret for classification as follows:
library(caret)
library(mlbench)
library(glmnet)
data(PimaIndiansDiabetes)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
fit.glmnet <- train(diabetes~., data=PimaIndiansDiabetes, method="glmnet",
                    metric="Accuracy", preProc=c("center", "scale"), trControl=trainControl)
# summarize fit
print(fit.glmnet)

# The glmnet implementation can be used in caret for regression as follows:
library(caret)
library(mlbench)
library(glmnet)
# Load the dataset
data(BostonHousing)
# train
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
fit.glmnet <- train(medv~., data=BostonHousing, method="glmnet", metric="RMSE",
                    preProc=c("center", "scale"), trControl=trainControl)
# summarize fit
print(fit.glmnet)


#### Part 2: Regression Trees and Model Trees -------------------

## Understanding regression trees and model trees ----
## Example: Calculating SDR ----
# set up the data
tee <- c(1, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7)
at1 <- c(1, 1, 1, 2, 2, 3, 4, 5, 5)
at2 <- c(6, 6, 7, 7, 7, 7)
bt1 <- c(1, 1, 1, 2, 2, 3, 4)
bt2 <- c(5, 5, 6, 6, 7, 7, 7, 7)

# compute the SDR
sdr_a <- sd(tee) - (length(at1) / length(tee) * sd(at1) + length(at2) / length(tee) * sd(at2))
sdr_b <- sd(tee) - (length(bt1) / length(tee) * sd(bt1) + length(bt2) / length(tee) * sd(bt2))

# compare the SDR for each split
sdr_a
sdr_b

## Example: Estimating Wine Quality ----
## Step 2: Exploring and preparing the data ----
wine <- read.csv("whitewines.csv")

# examine the wine data
str(wine)

# the distribution of quality ratings
hist(wine$quality)

# summary statistics of the wine data
summary(wine)

wine_train <- wine[1:3750, ]
wine_test <- wine[3751:4898, ]

## Step 3: Training a model on the data ----
# regression tree using rpart
library(rpart)
m.rpart <- rpart(quality ~ ., data = wine_train)

# get basic information about the tree
m.rpart

# get more detailed information about the tree
summary(m.rpart)

# use the rpart.plot package to create a visualization
library(rpart.plot)

# a basic decision tree diagram
rpart.plot(m.rpart, digits = 3)

# a few adjustments to the diagram
rpart.plot(m.rpart, digits = 4, fallen.leaves = TRUE, type = 3, extra = 101)

## Step 4: Evaluate model performance ----

# generate predictions for the testing dataset
p.rpart <- predict(m.rpart, wine_test)

# compare the distribution of predicted values vs. actual values
summary(p.rpart)
summary(wine_test$quality)

# compare the correlation
cor(p.rpart, wine_test$quality)

# function to calculate the mean absolute error
MAE <- function(actual, predicted) {
  mean(abs(actual - predicted))  
}

# mean absolute error between predicted and actual values
MAE(p.rpart, wine_test$quality)

# mean absolute error between actual values and mean value
mean(wine_train$quality) # result = 5.87
MAE(5.87, wine_test$quality)

## Step 5: Improving model performance ----
# train a M5' Model Tree
library(RWeka)
m.m5p <- M5P(quality ~ ., data = wine_train)

# display the tree
m.m5p

# get a summary of the model's performance
summary(m.m5p)

# generate predictions for the model
p.m5p <- predict(m.m5p, wine_test)

# summary statistics about the predictions
summary(p.m5p)

# correlation between the predicted and true values
cor(p.m5p, wine_test$quality)

# mean absolute error of predicted and true values
# (uses a custom function defined above)
MAE(wine_test$quality, p.m5p)
