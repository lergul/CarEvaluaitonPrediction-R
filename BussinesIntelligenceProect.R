# In this project, I will work on car evaluation prediction. First of all, the dataset I will use belongs to Marko Bohanec and Blaz Zupan. 
# We have 6 features, one target value and 1728 examples. 
# I will contiune step by step. 

# Step 0 Introduction
# I will import some library which are helpfull for us. I have missing some library that's why before the import 
# these library I installed that missing packages. 
# For installing you can use tools part or you can type this function >> install.packages()
install.packages("tidyverse")
install.packages("readr")
install.packages("ggplot2")
install.packages("caret")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("funModeling")
install.packages("cowplot")
install.packages("caTools")
install.packages("e1071")
install.packages("randomForest")
library(tidyverse)
library(readr)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(pander)
library(funModeling)
library(cowplot)
library(corrplot)
library(RColorBrewer)
library(caTools)
library(e1071)
library(tree)
library(randomForest)
# Step 1. Data Reading
# On this part I will upload dataset and looking more careful of them.
# I will look column names, missing values and giving summary about dataset.

cardata <- read.csv("car_evaluation.csv")

colnames(cardata)
## For better understanding I will change column names.

colnames(cardata) = c('Buying','Maint','Doors','Person','LugBoot','Safety','Evaluation')
colnames(cardata)
## Now we can understand more easily our features. 

#Checking missing values on each column 
colSums(is.na(cardata))

# Summary about dataset 
summary(cardata)
glimpse(cardata) #On dataset everything is categorical  glimpse function is a useful for string values.

# Step 2. Data Visualization

freq(cardata)

# Step 2.1
# On this step we will look how is the feature relations with Evaluation. 
# After the this visualization we can talk more clear about the which feature has an affect on Evaluation.

## Safety vs Evaluation
p1 <- ggplot(cardata, aes( Safety, fill = Evaluation )) +
  geom_bar(position = position_dodge()) + 
  ggtitle("Acceptability according to safety") +
  xlab("Safety") + 
  ylab("Observation Frequencies")

## Maintenance Cost vs Evaluation 
p2 <- ggplot(cardata, aes(Maint , fill = Evaluation )) +
  geom_bar(position = position_dodge()) + 
  ggtitle("Acceptability according to maintenance cost") +
  xlab("Maintenance Cost") + 
  ylab("Observation Frequencies")

# Person vs Evaluation
p3 <- ggplot(cardata, aes( Person , fill = Evaluation)) +
  geom_bar(position = position_dodge()) + 
  ggtitle("Acceptability according to capacity of person") +
  xlab("Capacity of Person") + 
  ylab("Observation Frequencies")

## Doors vs Evaluation 
p4 <- ggplot(cardata, aes(Doors, fill = Evaluation )) +
  geom_bar(position = position_dodge()) + 
  ggtitle("Acceptability according to number of doors") +
  xlab("Number of Doors") + 
  ylab("Observation Frequencies")

# LugBoot vs Evaluation
p5 <- ggplot(cardata, aes(LugBoot, fill = Evaluation )) +
  geom_bar(position = position_dodge()) + 
  ggtitle("Acceptability according to size of luggage boot") +
  xlab("The size of Luggage Boot") + 
  ylab("Observation Frequencies")

# Buying price vs Evaluation
p6 <- ggplot(cardata, aes(Buying, fill = Evaluation )) +
  geom_bar(position = position_dodge()) + 
  ggtitle("Acceptability according to buying price") +
  xlab("Buying Price") + 
  ylab("Observation Frequencies")

cowplot::plot_grid(p1, p2, p3, p4, p5, p6 ,labels = "AUTO")
# Step 2.2 Correlation
# On this part, I will look is there any relation between the each feaures.

cardata_cor <- cardata
for (i in 1:ncol(cardata_cor)){
  vf <- factor(cardata_cor[,i])
  as.character(vf)
  cardata_cor[i] <- as.numeric(vf)
}
res <- cor(cardata_cor)
round(res, 2)
corrplot(res, method = "number",type="full", order="hclust",
         col=brewer.pal(n=8, name="RdYlBu"))
## After the saw plotting result., there are not any relation between features. 6 feature just effect on target feature. As we except this one.
## Also after the train model we will look which feature has more important for our model. 

# Step 3 Machine Learning Model 
# First of all I will classify data set as a training set and test set.
cardata$Buying <- factor(cardata$Buying)
cardata$Maint <- factor(cardata$Maint)
cardata$Doors <- factor(cardata$Doors)
cardata$Person <- factor(cardata$Person)
cardata$LugBoot <- factor(cardata$LugBoot)
cardata$Safety <- factor(cardata$Safety)
cardata$Evaluation <- factor(cardata$Evaluation)



#First Data split: %90 training %10 test

#Second Data split: %70 training %30 test. 

#Third Data Split %50 training %50 Test

set.seed(1234)
training <- createDataPartition(cardata$Evaluation, p = .70,
                                list = FALSE,
                                times = 1)
train <- cardata[training, ]
test <- cardata[-training, ]

train_x <- train %>% dplyr::select(-Evaluation)
train_y <- train$Evaluation
test_x <- test %>% dplyr::select(-Evaluation)
test_y <- test$Evaluation
# distribution of Attrition rates across train & test set
table(train$Evaluation) %>% prop.table() 
##
table(test$Evaluation) %>% prop.table()

##Model 1: Classification Tree Model

#fit the tree model using training data 
tree_model = tree(Evaluation~., train)
plot(tree_model)
text(tree_model,pretty = 0)
#check how the model is doing using the test data

fit1=rpart(formula = Evaluation~ (Safety+Buying+Person+LugBoot+Doors+Maint),
           data=train,method = "class")
rpart.plot(fit1)
text(fit1, pretty = 0)

tree_pred = predict(tree_model, test, type = "class")
cfm_tree <- confusionMatrix(tree_pred, test$Evaluation)
cfm_tree
acc_tree <- 1-mean(tree_pred != test$Evaluation) ## It shows us the accuracy of the model

## Model 2: Naive Bayes
car.nb <- naiveBayes(Evaluation ~. , data = train )
car.nb_predict <- predict(car.nb, test[ , names(test) != "Evaluation"])
cfm <- confusionMatrix(car.nb_predict, test$Evaluation)
cfm
acc_nb <- 1-mean(car.nb_predict != test$Evaluation) # It shows us the accuracy of the model
# Compare between the two classification approaches
compare = rbind(acc_tree, acc_nb)
if (acc_tree > acc_nb){
  print("The Decision Trees approach is more accurate than the Naive Bayes one")
  pandoc.table(compare)
} else if (acc_tree < acc_nb){
  print("The Naive Bayes approach is more accurate than the Decision Trees one")
  pandoc.table(compare)
} else {
  print("The two classification approaches have the same accuracy level")
  pandoc.table(compare)
}


## K-fold Cross Validation with Naive Bayes and Decision Tree

# create response and feature data
features <- setdiff(names(cardata), "Evaluation")
x <- cardata[, features]
y <- cardata$Evaluation

# set up 10-fold cross validation procedure
train_control <- trainControl(
  method = "cv", 
  number = 10
)

# train model
nb.m1 <- train(
  x = x,
  y = y,
  method = "nb",
  trControl = train_control
)
nb.m1
# results
confusionMatrix(nb.m1)


# Decision Tree Model K-fold Cross validation

features <- c("Buying", "Maint","Doors","Person","LugBoot","Safety","Evaluation")
# Set up caret to perform 10-fold cross validation repeated 3 times
caret.control <- trainControl(method = "repeatedcv",
                              number = 10)
dec.tree <- train(Evaluation ~ .,
                   data = cardata[, features],
                   method = "rpart",
                   trControl = caret.control,
                   tuneLength = 15
)
dec.tree

# results
confusionMatrix(dec.tree)
#fit the tree model using training data 
 
## Random forest used for the importance feature
set.seed(2376)
model1 <- randomForest(train_x, train_y,importance = TRUE)
model1
importance(model1)
varImpPlot(model1)


### Predictions
defaultSummary(data.frame(obs = test_y,
                          pred = predict(model1,test_x)))


