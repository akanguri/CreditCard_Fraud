
#Library used
library(caTools)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(ggthemes)

# Set present working directory
setwd('/Users/Ameet/Box Sync/Ameet/Classes/R/')

# Read data from Credit card
data=read.csv('creditcard.csv')
colnames(data)[31] ='Fraud'
data$Fraud=as.factor(data$Fraud)

# Structure of data
str(data)
names(data)
table(data$Class)
# data$Fraud=ifelse(data$Class==1,'Yes','No')

# Compare the Time and amount spend to see of there anything significant
ggplot(data=data,aes(x=Time,y=Amount))+
  geom_point(alpha=1,colour = 'blue', size=1)+
  facet_grid(Class~.,scales = "free_y")+
  theme_solarized()



# hard to see any pattern apart from the fact that All Fraud transactions are lesser in value

# Plot a histogram on each of the features to see if there are any visual patterns

columns= names(data[,2:29])
for (col in columns){
  print(ggplot(data = data, aes(x = data[,col], fill = Class)) + 
          geom_histogram(alpha = 0.5, bins = 50,position = 'identity',aes(y = ..density..))+
          scale_fill_manual(values=c("#66CC99","#FF9999"))+
          labs( x = col)+
          geom_density(alpha = 0.5, inherit.aes = FALSE, aes(x=data[,col],colour=Class), data = data)+
          scale_color_manual(values=c("#66CC99","#FF9999")))
}

# Going through the vizualizations, certain features have similiar distributions of data across both normal and fraud transactions.
# We can eliminate variables and focus on ones which show variations in patterns
dropCol=c('V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8')
data= data[,! names(data) %in% dropCol]


subPositive=subset(data, data$Class == 0)
subNegative=subset(data, data$Class == 1)
subPositive=subPositive[sample(nrow(subPositive),492,replace = FALSE),]
dataSample=rbind(subNegative,subPositive)
dataSample=dataSample[sample(nrow(dataSample)),]

# Split data into train and test
split =sample.split(dataSample$Class,SplitRatio = 0.7)
train=dataSample[split,]
test=dataSample[!split,]

table(train$Class)
table(test$Class)

#
model1=glm(train$Fraud~.,data = train,family ='binomial')
predModel=predict(model1,test,type="response")
ct = table(test$Fraud,predModel>.98); ct
accuracy = sum(ct[1,1],ct[2,2])/nrow(test); accuracy
specificity = ct[1,1]/sum(ct[1,1],ct[1,2]); specificity
sensitivity = ct[2,2]/sum(ct[2,1],ct[2,2]); sensitivity

# Construct a default CART Accuracy -93.5%, Specificity-96.6%, Sensitivity-90.5%

tree = rpart(Fraud~.,data=train, method = 'class')
predTree = predict(tree,newdata=test,type='class')
ct = table(test$Fraud,predTree); ct
accuracy = sum(ct[1,1],ct[2,2])/nrow(test); accuracy
specificity = ct[1,1]/sum(ct[1,1],ct[1,2]); specificity
sensitivity = ct[2,2]/sum(ct[2,1],ct[2,2]); sensitivity


# Construct a maximal tree Accuracy -93,5%, Specificity-96.6%, Sensitivity-90.5%
maximalTree = rpart(Fraud~.,data=train,control=rpart.control(minbucket=1))
predMaximalTree = predict(maximalTree,newdata=test)
ct = table(test$Fraud,predTree); ct
accuracy = sum(ct[1,1],ct[2,2])/nrow(test); accuracy
specificity = ct[1,1]/sum(ct[1,1],ct[1,2]); specificity
sensitivity = ct[2,2]/sum(ct[2,1],ct[2,2]); sensitivity


## Use cross-validation to prune the tree Accuracy -91.2%, Specificity-97.9%, Sensitivity-84.4%
library(caret); library(e1071)
set.seed(100)
trControl = trainControl(method="cv",number = 10)
tuneGrid = expand.grid(.cp = seq(0.001,0.1,0.001))
cvModel = train(Fraud~.,data=train,method="rpart",
                trControl = trControl,tuneGrid = tuneGrid)
treeCV = rpart(Fraud~.,data=train,
               control=rpart.control(cp = cvModel$bestTune))
predTreeCV = predict(treeCV,newdata=test)
ct = table(test$Fraud,predTreeCV[,2]>.98); ct
accuracy = sum(ct[1,1],ct[2,2])/nrow(test); accuracy
specificity = ct[1,1]/sum(ct[1,1],ct[1,2]); specificity
sensitivity = ct[2,2]/sum(ct[2,1],ct[2,2]); sensitivity



### Bagging
### Bagging Packages: randomForest,adabag, bagEarth, treeBag, bagFDA

### Bagging using randomForest with mtry=ncol(train)-1
# Accuracy -92%, Specificity-95.9%, Sensitivity-89.8%
library(randomForest)
set.seed(100)
bag = randomForest(Fraud~.,data=train,mtry = ncol(train)-1,ntree=1000)
predBag = predict(bag,newdata=test)

ct = table(test$Fraud,predBag); ct
accuracy = sum(ct[1,1],ct[2,2])/nrow(test); accuracy
specificity = ct[1,1]/sum(ct[1,1],ct[1,2]); specificity
sensitivity = ct[2,2]/sum(ct[2,1],ct[2,2]); sensitivity

plot(bag)
varImpPlot(bag); importance(bag)  ## see variable importance
getTree(bag,k=100)   # View Tree 100
hist(treesize(bag))  # size of trees constructed 


## Random Forest Accuracy -94.5%, Specificity-98.6%, Sensitivity-90.5%
set.seed(100)
forest = randomForest(Fraud~.,data=train,ntree = 1000)
predForest = predict(forest,newdata=test)

ct = table(test$Fraud,predForest); ct
accuracy = sum(ct[1,1],ct[2,2])/nrow(test); accuracy
specificity = ct[1,1]/sum(ct[1,1],ct[1,2]); specificity
sensitivity = ct[2,2]/sum(ct[2,1],ct[2,2]); sensitivity


names(forest)
summary(forest)
plot(forest)
varImpPlot(forest); importance(forest)  ## see variable importance
getTree(forest,k=100)   # View Tree 100
hist(treesize(forest))  # size of trees constructed 

## Random Forest picks a default value of mtry if one is not given
## Using cross-validation to pick the optimal mtry
## This may take a while to run
#  Accuracy -94.2%, Specificity-98.6%, Sensitivity-89.86%
trControl=trainControl(method="cv",number=10)
tuneGrid = expand.grid(mtry=1:5)
set.seed(100)
cvForest = train(Fraud~.,data=train,
                 method="rf",ntree=1000,trControl=trControl,tuneGrid=tuneGrid )
cvForest  # best mtry was 2
set.seed(100)
forest = randomForest(Fraud~.,data=train,ntree = 1000,mtry=2)
predForest = predict(forest,newdata=test)
predForest1 = predict(forest,newdata=test,type ='prob')

ct = table(test$Fraud,predForest1[,2]>.5); ct
accuracy = sum(ct[1,1],ct[2,2])/nrow(test); accuracy
specificity = ct[1,1]/sum(ct[1,1],ct[1,2]); specificity
sensitivity = ct[2,2]/sum(ct[2,1],ct[2,2]); sensitivity


## Boosting
# Many boosting models available
## gbm: Boosting with trees
## mboost: Model based boosting
## ada: statistical boosting based on addditive logistic regression
## gamBoost: boosting GAM
#  Accuracy -93.3%, Specificity-98.6%, Sensitivity-89.1%
library(gbm)
set.seed(100)
boost = gbm(ifelse(train$Fraud==1,1,0)~.,data=train,
            n.trees = 10000,interaction.depth = 3,shrinkage = 0.001,distribution = 'bernoulli')

predBoost = predict(boost,newdata=test,n.trees = 10000)


ct = table(test$Fraud,predBoost>.98); ct
accuracy = sum(ct[1,1],ct[2,2])/nrow(test); accuracy
specificity = ct[1,1]/sum(ct[1,1],ct[1,2]); specificity
sensitivity = ct[2,2]/sum(ct[2,1],ct[2,2]); sensitivity

### With boosting, performance can vary widely based on parameters provided
## Warning: Tuning can take a long while
## Note, I have not tuned for n.trees because it takes too long
## Also, I have used a small number for n.trees. For boosting a larger number
# is likely to help but also take longer to process.
# Accuracy -94%, Specificity-98%, Sensitivity-91.2%
set.seed(100)
trControl=trainControl(method="cv",number=10)
#tuneGrid = expand.grid(n.trees=(1:10)*1000, shrinkage=0.01,interaction.depth=2)
tuneGrid=  expand.grid(n.trees = 1000, interaction.depth = c(1,2),
                       shrinkage = (1:100)*0.001, n.minobsinnode=10)
cvBoost = train(Fraud~.,data=train,method="gbm", 
                trControl=trControl, tuneGrid=tuneGrid)
boostCV = gbm(Fraud~.,data=train,distribution="bernoulli",
              n.trees=1000,interaction.depth=2,shrinkage=0.014)
predBoostCV = predict(boostCV,test,n.trees=1000)

ct = table(data$Fraud,predBoostCV>0); ct
accuracy = sum(ct[1,1],ct[2,2])/nrow(data); accuracy
specificity = ct[1,1]/sum(ct[1,1],ct[1,2]); specificity
sensitivity = ct[2,2]/sum(ct[2,1],ct[2,2]); sensitivity

predTreeProb = predict(tree,newdata=test,type="prob")
ROCRpred = prediction(predTreeProb[,2],test$violator)
as.numeric(performance(ROCRpred,"auc")@y.values) # auc measure
## construct plot
ROCRperf = performance(ROCRpred,"tpr","fpr")
plot(ROCRperf,colorize=TRUE,print.cutoffs.at=seq(0,1,0.2),text.adj=c(-0.3,2),xlab="False Positive Rate (1 - Specificity) ",ylab="True Positive Rate (Sensitivity)") # color coded and annotated ROC curve


library(ROCR)
ROCRpred = prediction(predForest1[,2],data$Fraud)
as.numeric(performance(ROCRpred,"auc")@y.values) # auc measure
## construct plot
ROCRperf = performance(ROCRpred,"tpr","fpr")
plot(ROCRperf,colorize=TRUE,print.cutoffs.at=seq(0,1,0.2),text.adj=c(-0.3,2),xlab="False Positive Rate (1 - Specificity) ",ylab="True Positive Rate (Sensitivity)") # color coded and annotated ROC curve


install.packages("PRROC")

library(PRROC)
fg <- predForest1[,2]
bg <- predForest1[,1]

# ROC Curve    
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(roc)

# PR Curve
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)



