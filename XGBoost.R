
setwd("~/Documents/OGGIE/Dissertation/Stage_4/1.6_Boosting")
test1 <- read.table("ProcessedData.csv", sep=",", header=T, na.strings=c("",".","NA")) # log norm 
test1$Cancer <- as.factor(test1$Cancer)

### split to test-train
library(caret)
inTraining <- createDataPartition(test1$Cancer, p = .75, list = FALSE)
training <- test1[ inTraining,]
testing  <- test1[-inTraining,]

#################################
##### test2                  ####    
#####REMOVE HIGHLY COLLINEAR ####
#################################
library(caret)

RemoveCollinear <- function(dataIn){
    target <- 'Cancer'
    predictors <- setdiff(names(dataIn), target)
    # Remove highly correlated predictors
    train_corr <- cor(dataIn[ ,predictors])
    train_high_corr_v <- findCorrelation(train_corr, cutoff=.8)
    train_low_corr <- dataIn[ ,-c(train_high_corr_v)]
    dataOut <- train_low_corr
    print(paste0("Number of more than 80% correlated columns : ", length(train_high_corr_v)))
    return(dataOut)
}

test2 <- RemoveCollinear(test1)
test2
keep <- names(test2[, -which(names(test2) %in% c("Cancer"))])
##########################################
#### Test 3 ----> males only variable selected  ##
##########################################
test3 <- test1[test1[, "Sex"] == 1,]
drops <- c("Sex")
test3 <- test3[ , !(names(test3) %in% drops)]
test3 <- RemoveCollinear(test3)
keep <- names(test3[, -which(names(test3) %in% c("Cancer"))])
test3



#########################
#### Test 4 - sffs
########################
str(test1)
library(dprep)
sffs(test1)
# penalized SVM -
# The R package ‘penalizedSVM’ provides two wrapper feature selection methods 

#####################################################################
# Test5 - Filter methods, e.g. Recursive Feature Elimination (RFE) (
######################################################################
# https://www.r-bloggers.com/introduction-to-feature-selection-for-bioinformaticians-using-r-correlation-matrix-filters-pca-backward-selection/
# specify and run rfe
library(mlbench)
library(caret)
set.seed(1)


library(caret);
#load caret library

keep <- names(test1[, -which(names(test1) %in% c("Cancer"))])
data_features<-as.matrix(test1[keep])
#load data features

data_class<-as.matrix(test1[3]);
#load data classes

data_features<- scale(data_features, center=TRUE, scale=TRUE);
#scale data features

inTrain <- createDataPartition(data_class, p = 3/4, list = FALSE); 
#Divide the dataset in train and test sets

#Create the Training Dataset for Descriptors 
trainDescr <- data_features[inTrain,];

# Create the Testing dataset for Descriptors
testDescr <- data_features[-inTrain,];

trainClass <- data_class[inTrain];
testClass <- data_class[-inTrain];


descrCorr <- cor(trainDescr);
highCorr <- findCorrelation(descrCorr, 0.9);
trainDescr <- trainDescr[, -highCorr];
testDescr <- testDescr[, -highCorr];
# Here, we can included a correlation matrix analysis to remove the redundant features before the backwards selection 

svmProfile <- rfe(x=trainDescr, y = trainClass, sizes = c(1:5), rfeControl= rfeControl(functions = caretFuncs,number = 2),method = "svmRadial",fit = FALSE);

svmProfile <- rfe(x=trainDescr, y = trainClass, sizes = c(1:5), rfeControl= rfeControl(functions = rfFuncs,number = 2),fit = FALSE);
# random forest

#caret function: the rfe is the backwards selection, c is the possible sizes of the features sets, and method the optimization method is a support vector machine.
svmProfile$variables
summary(svmProfile)
svmProfile$results
svmProfile$bestSubset
chosenVar <- svmProfile$variables$var
test5 <- test1[,chosenVar]
dim(test5) #only 122 features left
dim(test1)
svmProfile$variables$var
###??? more variables now..


#varSelRF package



### XGBoost
library(xgboost)

### XGBoost function
### parameters: 
# trainIn = training set
# testIn = test set
# maxDepthIn = maximum depth of tree
# positive_WeightIn = weight of positive class range {0.000000000000001 --> 9999999999999999}

keep <- names(training[, -which(names(training) %in% c("Cancer"))])
fit_XGB <- function(trainIn, testIn, maxDepthIn, positive_WeightIn){
    X <- as.matrix(trainIn[keep])
    numeric_X <- apply(X, 2, as.numeric)
    Y <- as.matrix(trainIn$Cancer)
    numeric_Y <- as.numeric(Y)
    model <- xgboost(data = numeric_X, label = numeric_Y,
                     nrounds = 10, max_depth=maxDepthIn, objective = "binary:logistic", 
                     scale_pos_weight = positive_WeightIn, nthread = 4)  # make scale_pos_weight --> small = all 0's, increase it to a 1000, = more 1's #sumwneg / sumwpos.  #max_depth = 2 #set_weight(weight) 
    Xtest <- as.matrix(testIn[keep])
    numeric_Xtest <- apply(Xtest, 2, as.numeric)
    predictions <- predict(model, numeric_Xtest, type="prob")
    return(predictions)
}





#### fit boot strap --> downsampled--> XGboost
bootstrap_downsample_fit_XGB <- function(trainIn, testIn, maxdepthIn, num_boots = 30) {
    keep <- names(trainIn[, -which(names(trainIn) %in% c("Cancer"))])
    
    prediction_collect <- matrix(NA, nrow = length(testIn$Cancer), ncol = num_boots)
    for (i in 1:num_boots){
        # bootstrap sample with replacement
        index <- sample(1:nrow(trainIn), 100, replace = TRUE) 
        boot_Sample <- trainIn[index, ]
        # undersample
        down_train <- downSample(x = boot_Sample[keep],
                                 y = boot_Sample$Cancer)
        colnames(down_train)[colnames(down_train)=="Class"] <- "Cancer"
        pred_proba <- fit_XGB(down_train, testIn, maxDepthIn=maxdepthIn, positive_WeightIn=1)
        prediction_collect[,i] <- pred_proba
    }
    MajorityVote <- rowSums(prediction_collect)/num_boots
    return(MajorityVote)
}
ew <- bootstrap_downsample_fit_XGB(training, testing, maxdepthIn = 10, num_boots = 20)
ew2 <- fit_XGB(training, testing, maxDepthIn = 20, positive_WeightIn = 1)




##########################################
###### Optimizing SMOTE parameters #######
##########################################

percOver <- c(50, 75, 100, 125, 150, 175, 200, 300, 400)
percUnder <- c(50, 75, 100, 125, 150, 175, 200, 300, 400)
kList <- c(2,3,4,5,6,7,8,9,10)

library(rpart)
### Optimize SMOTE via CART with default parameters


#matrix: FNR, FPR, percOver, percUnder, k
me.matrix <- matrix(NA, nrow=10000, ncol= 5) #and truncate it at the end.
for (i in 1:length(percOver)){
    x <- percOver[i]
    for (j in 1:length(percUnder)){
        y <- percUnder[j]
        for (k in 1:length(kList)){
            kk <- kList[k]
            smoted <- SMOTE(Cancer ~ ., training, perc.over = x, perc.under=y, k=kk)
            smoted_model <- rpart(Cancer ~ .,data=smoted,parms=list(prior=c(.5,.5)))
            predictions <- predict(smoted_model, testing, probability=TRUE)
            pred_proba <- as.numeric(predictions[1:(length(predictions)/2),2])
            ress <- returnThreshedVals(pred_proba, 0.5)
            FNR_ans <- returnMetric(ress, testing$Cancer, "FNR")
            FPR_ans <- returnMetric(ress, testing$Cancer, "FPR")
            NAindex <- which(is.na(me.matrix))
            firstNA <- as.integer(min(NAindex))
            me.matrix[firstNA, 1] <- FNR_ans
            me.matrix[firstNA, 2] <- FPR_ans
            me.matrix[firstNA, 3] <- x 
            me.matrix[firstNA, 4] <- y 
            me.matrix[firstNA, 5] <- kk
        }
    }
}

doo <- data.frame(me.matrix)
dee <- na.omit(doo)
colnames(dee) <- c("FNR", "FPR", "per.Over", "per.Under", "K")
dee

roundedData <- round_df(dee, 2)
roundedData
outty <- subset(roundedData, FNR < 0.05 & FPR <0.3)

write.table(outty, file = "grid_smote_TestTrain_rpart.txt", sep = "\t", row.names = FALSE)

#### SMOTE PARAMETERS
#FNR	FPR	per.Over	per.Under	K
#0	0.25	50	150	10
#0	0.21	75	150	10
#0	0.27	75	400	10
#0	0.25	100	50	10
#0	0.15	150	125	10
#0	0.29	150	175	10
#0	0.21	200	150	10
#0	0.27	300	50	10


library(DMwR)
SMOTED_fit_XGB <- function(trainIn, testIn, maxdepthIn, num_boots = 30) {
    #prediction_collect <- matrix(NA, nrow = length(testIn$Cancer), ncol = num_boots)
    prediction_collect <- matrix(NA, nrow = length(testIn$Cancer), ncol = num_boots)
    for (i in 1:num_boots){
        # smote it
        smotey <- SMOTE(Cancer ~ ., trainIn, perc.over = 150, perc.under=125, k = 10)
        pred_proba <- fit_XGB(smotey, testIn, maxDepthIn=maxdepthIn, positive_WeightIn=1)
        
        prediction_collect[,i] <- pred_proba
    }
    MajorityVote <- rowSums(prediction_collect)/num_boots
    return(MajorityVote)
}
dee <- SMOTED_fit_XGB(training, testing, maxdepthIn=20, num_boots = 30)
dee







#### Return FNR
### predictionsIN: vector of predictions
### testIn: testSet
### returnWhat --> define metric to return - either FNR, FPR
returnMetric <- function(predictionsIn, testIn, returnWhat) {
    cm <- table(predictionsIn, testIn)
    dimension <- dim(cm)
    
    cm <- t(cm)
    cm_row <- dimension[1]
    if (cm_row ==1) {
        attrit <- attributes(cm)
        predicted_level <- attrit$dimnames$predictionsIn
        if(predicted_level == "0"){
            TN <- cm[1]
            FN <- cm[2]
            TP <- 0
            FP <- 0
            if (returnWhat=="FNR"){
                FNR <-  FN / (FN + TP) 
                #print(FNR)
                return(FNR)
            }
            if (returnWhat == "FPR"){
                FPR <- FP / (FP + TN)
                #print(FPR)
                return(FPR)
            }
            
        }
        if(predicted_level =="1") {
            FP <- cm[1]
            TP <- cm[2]
            TN <- 0
            FN <- 0
            if (returnWhat=="FNR"){
                FNR <-  FN / (FN + TP) 
                #print(FNR)
                return(FNR)
            }
            if (returnWhat == "FPR"){
                FPR <- FP / (FP + TN)
                #print(FPR)
                return(FPR)
            }
        }
    }
    if (cm_row == 2) { 
        TN <- cm[1]
        FN <- cm[2]
        FP <- cm[3]
        TP <- cm[4]
        if (returnWhat=="FNR"){
            FNR <-  FN / (FN + TP) 
            #print(FNR)
            return(FNR)
        }
        if (returnWhat == "FPR"){
            FPR <- FP / (FP + TN)
            #print(FPR)
            return(FPR)
        }
        #TPR <- TP / (TP + FN)
        #print(TPR)
    }
}


returnThreshedVals <- function(vectorIn,cutOffIn){
    pred_proba <- vectorIn 
    pred_proba <- ifelse(pred_proba>cutOffIn,1,0) # change cut off point
    return(pred_proba)
}
returnThreshedVals(dee, cutOffIn = 0.5)


library("pROC")

KFoldCV_XGB <- function(dataIn, cutIndex,depthIn, positive_WeightIn, threshIn){
    cancerYes <- dataIn[dataIn[, "Cancer"] == 1,]
    cancerNo <- dataIn[dataIn[, "Cancer"] == 0,]
    
    # randomly shuffle data 
    cancerYes <- cancerYes[sample(nrow(cancerYes)),]    
    cancerNo <- cancerNo[sample(nrow(cancerNo)),]
    
    FNR_collect = c()
    TPR_collect = c()
    #average_AUC_score = c()
    for (i in 1:max(unique(cutIndex[[1]]))) {
        testIndexes_pos <- which(cutIndex[[1]]==i,arr.ind=TRUE)
        testIndexes_neg <- which(cutIndex[[2]]==i,arr.ind=TRUE)
        #print(testIndexes_pos)
        
        testPos <- cancerYes[testIndexes_pos,]
        testNeg <- cancerNo[testIndexes_neg,]
        trainPos  <- cancerYes[-testIndexes_pos,]
        trainNeg <- cancerNo[-testIndexes_neg,]
        
        testSet <- rbind(testPos, testNeg)
        trainSet <- rbind(trainPos, trainNeg)
        
        # Normal XGBoost
        #K_pred <- fit_XGB(trainSet, testSet, depthIn, positive_WeightIn)
        
        # Bootstrap --> downsample --> XGboost
        #K_pred <- bootstrap_downsample_fit_XGB(trainSet, testSet, maxdepthIn = 30, num_boots = 30)
        
        # SMOTE --> XGboost
        
        K_pred <- SMOTED_fit_XGB(trainSet, testSet, maxdepthIn=20, num_boots = 30)
        
        predictions <- returnThreshedVals(K_pred,  threshIn)
        
        print(predictions)
        print(testSet$Cancer)

        FNRates <- returnMetric(predictions, testSet$Cancer, "FNR")
        FPRates <- returnMetric(predictions, testSet$Cancer, "FPR")
        
        FNR_collect <- c(FNR_collect,FNRates)
        TPR_collect <- c(TPR_collect,FPRates)
    }
    mean_FNR <- mean(FNR_collect)
    mean_TPR <- mean(TPR_collect)
    
    return(list=c(mean_FNR,mean_TPR)) # return average FNR, FPR
}


CutData <- function(dataIn, Kcut) {
    cancerYes <- dataIn[dataIn[, "Cancer"] == 1,]
    cancerNo <- dataIn[dataIn[, "Cancer"] == 0,]
    
    cancerYes <- cancerYes[sample(nrow(cancerYes)),] # randomly shuffle data    
    cancerNo <- cancerNo[sample(nrow(cancerNo)),]
    folds1 <- cut(seq(1,nrow(cancerYes)),breaks=Kcut,labels=FALSE) 
    folds2 <- cut(seq(1,nrow(cancerNo)),breaks=Kcut,labels=FALSE) 
    mylist <- list(folds1, folds2)
    return(mylist)   
}


test1Cuts <- CutData(test1, Kcut = 5)
KFoldCV_XGB(test1, cutIndex = test1Cuts, depthIn =  10, positive_WeightIn=1, threshIn = 0.5)


#### run k fold cross validation many times
multiple_K_folds <- function(dataIn, cutsIn, depthIn, positive_WeightIn, threshIn) {
    FNR_collect <- c()
    FPR_collect <- c()
    for (i in 1:10) {
        TheCuts <- CutData(dataIn, cutsIn)
        #stats_ <- KFoldCV_SVM(dataIn=dataIn, cutIndex = TheCuts, weightIn = weightIn, costParamIn = costParamIn, gammaParamIn = gammaIn, threshIn = threshIn)
        #stats_ <-KFoldCV_TREE(dataIn=dataIn, cutIndex = TheCuts, weightIn = weightIn, cpIn=cpIn,  minsplitIn = minsplitIn, maxdepthIn = maxdepthIn, GiniInfo =  GiniInfo, threshIn = threshIn)
        stats_ <- KFoldCV_XGB(dataIn, cutIndex = TheCuts, depthIn = depthIn, positive_WeightIn=positive_WeightIn, threshIn = threshIn)
        
        FNR <- stats_[1]
        FPR <- stats_[2]
        FNR_collect <- c(FNR_collect, FNR)
        FPR_collect <- c(FPR_collect, FPR)
    }
    FNR_out <- mean(FNR_collect) 
    FPR_out <- mean(FPR_collect) 
    return(list(FNR_out, FPR_out )) 
} 

#depth 5 + depth 20
multiple_K_folds(dataIn=test1, cutsIn = 7, depthIn = 20, positive_WeightIn = 1, threshIn = 0.3) 
#0.2 PRODUCE FPR of 0.93
#0.6 produce too high FNR of 0.48
# threshold of 0.4 produced 0.05 FNR
# 0.5 too high




set.seed(777)
# loop through thresholds, return FPR, FNR
# plot
The_costs <- table(testing$Cancer)  # the weight vector must be named with the classes names
The_costs[1] <- 0.5
The_costs[2] <- 0.5


#multiple_K_folds(dataIn=test1, cutsIn = 7, depthIn = 3, positive_WeightIn = 1, threshIn = 0.05)
plotFPR_FNR <- function(positive_WeightIn, depthIn, Colour, first) {
    FNRcollect = c()
    FPRcollect = c()
    #thresh_value_list <- c(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1)
    
    # ORIGINAL XGBOOST SET OF THRESHOLDS
    #thresh_value_list <- c(0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05)
    
    #thresh_value_list <- c(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1)
    # NEW BOOT XGBOOST SET OF THRESHOLDS
    thresh_value_list <- c(0.3, 0.325, 0.35, 0.375, 0.4)
    
    for (i in 1:length(thresh_value_list)){
        thresh_cut <- thresh_value_list[i]
        #out <- multiple_K_folds(dataIn=test2, cutsIn = 3, weightIn = weightsIn, costParamIn = costIn, gammaIn=gammaIn, threshIn = thresh_cut)
        #out <- multiple_K_folds(dataIn=test2, cutsIn = 5, weightIn = weightsIn, cpIn=cpIn,  minsplitIn = minsplitIn, maxdepthIn = maxdepthIn,GiniInfo =  GiniInfo, threshIn = thresh_cut)
        out <- multiple_K_folds(dataIn=test2, cutsIn = 5, depthIn = depthIn, positive_WeightIn = positive_WeightIn, threshIn = thresh_cut)
        FNRout <- out[[1]]
        FPRout <- out[[2]]
        FNRcollect <- c(FNRcollect, FNRout)
        FPRcollect <- c(FPRcollect, FPRout)
    }
    if (first==TRUE) {
        #plot(FNRcollect, FPRcollect, type="l", main="XGBoost", col=Colour, ylim=range(0:1), xlim=range(0:0.05), xlab="FNR", ylab="FPR") #xlim=range(-0.5:0.5)
        #grid(NULL, NULL)
        
        #
        plot(FNRcollect, FPRcollect, type="o", main="SMOTE [k=10, perc.over = 150, perc.under=125] \n --> XGBoost", col=Colour, axes=FALSE, frame.plot = TRUE, xlab="FNR", ylab="FPR", ylim = range(0:1)) #xlim=range(-0.5:0.5)
        
        # visualize - setting xlim and ylim for small decimal places. 
        my.Xat <- c(0.01, 0.02, 0.03, 0.04, 0.05)
        axis(1, at = my.Xat, labels = my.Xat)
        my.yat <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
        axis(2, at = my.yat, labels = my.yat)
        
        # visualize: fine tune the grid 
        for (i in seq(0,0.1,0.01)){
            i2 <- i * 10
            abline(h = i2, v = i, col="lightgray", lty="dotted") 
        }
        
        
    }
    else {
        lines(FNRcollect, FPRcollect, type="o", col=Colour)
    }
    
    Majority_Class_Weight <- as.data.frame(rep(positive_WeightIn,length(thresh_value_list)))
    FNRf <- as.data.frame(FNRcollect)
    FPRf <- as.data.frame(FPRcollect)
    ThreshF <- as.data.frame(thresh_value_list)
    output <- cbind(FNRf, FPRf, ThreshF,   Majority_Class_Weight)
    colnames(output) <- c("FNR", "FPR", "CutOff-thresh", "Majority Class Weight")
    return(output)
}

results <- plotFPR_FNR(positive_WeightIn = 1, depthIn = 20, "black", first= TRUE)

roundedData <- round_df(results, 3)
outty <- subset(roundedData, FNR < 0.05 & FPR <0.7)
outty




#### Create a matrix of weight parameters
The_costs <- table(testing$Cancer)  # the weight vector must be named with the classes names
The_costs[1] <- 0.5
The_costs[2] <- 0.5
rowNum <- 30
mymatrix <- matrix(NA, nrow=rowNum, ncol=2)
mymatrix[1,1] <- The_costs[1]
mymatrix[1,2] <- The_costs[2]

for (i in 1:(rowNum-1)){
    ind <- i+1
    The_costs[1] <- The_costs[1]/2 
    The_costs[2] <- The_costs[2]*2
    mymatrix[ind,1] <- The_costs[1]
    mymatrix[ind,2] <- The_costs[2]
}
mymatrix
### for every 4 indexes, create a roc curve
The_costs <- table(test1$Cancer)  # the weight vector must be named with the classes names
The_costs[1] <- 0.5
The_costs[2] <- 0.5

firstData <- plotFPR_FNR(positive_WeightIn = The_costs[2], depthIn = 30, "black", first= TRUE)

#firstData <- plotFPR_FNR(weightsIn=The_costs, cpIn = 0.1, minsplitIn = 25, maxdepthIn = 7, GiniInfo = "gini", "black", first= TRUE)
#plotFPR_FNR(weightsIn=The_costs, cpIn = 0.5, minsplitIn = 5, maxdepthIn = 10, GiniInfo = "gini", "black", first= TRUE)


Mycolour <- c("blue", "green", "orange", "purple", "red")
for (i in 1:(nrow(mymatrix)/4)) {
    rowInd <- i * 4
    weight0 <- mymatrix[rowInd,1]
    weight1 <- mymatrix[rowInd,2]
    print(weight1)
    The_costs[1] <- weight0
    The_costs[2] <-weight1
    #moreData <- plotFPR_FNR(weightsIn=The_costs[1], cpIn = 0.1, minsplitIn = 25, maxdepthIn = 7, GiniInfo = "gini", Mycolour[i], first= FALSE)
    moreData <- plotFPR_FNR(positive_WeightIn = The_costs[2], depthIn = 30, Mycolour[i], first= FALSE)
    firstData <- rbind(firstData, moreData)
    #    plotFPR_FNR(weightsIn=The_costs, cpIn = 0.5, minsplitIn = 5, maxdepthIn = 10, GiniInfo = "gini", Mycolour[i], first= FALSE)
}
mymatrix

# Add a legend
# legend(x_coordinate, y_coordinate)
legend(0.025, 1, legend=c("0.5", "4", "64", "1024", "16384", "262144"),
       col=c("black", "blue", "green", "orange", "purple", "red"), lty=1:1:1:1:1:1, cex=0.8,  box.lty=0, title="C+ weight")

firstData
#round numeric values in dataframe to 3 digits
round_df <- function(x, digits) {
    numeric_columns <- sapply(x, mode) == 'numeric'
    x[numeric_columns] <-  round(x[numeric_columns], digits)
    x
}

roundedData <- round_df(firstData, 2)
roundedData
outty <- subset(roundedData, FNR < 0.05 & FPR <0.37)
outty

outty <- subset(roundedData, FNR < 0.01 & FPR <0.41)
outty

##### CLASS WEIGHT (x) vs FPR
# cut off threshold - 0.04 
# FNR




###################################
#### FPR vs weights
###################################


#threshold 0.04
#class weight - 0.5 ...
##    0.5

positive_WeightIn_list <- c(0.03625, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8)

#positive_WeightIn_list <- c(15, 30, 60, 125, 250, 500, 1000)

FPR_collect <- c()
FNR_collect <- c()


for (i in 1:length(positive_WeightIn_list)){
    posWeight <- positive_WeightIn_list[i]
    out <- multiple_K_folds(dataIn=test2, cutsIn = 5, depthIn = 30, positive_WeightIn = posWeight, threshIn =0.04)
    
#    out <- multiple_K_folds(dataIn=test2, cutsIn = 5, depthIn = 30, positive_WeightIn = posWeight, threshIn =0.04)
    FNRout <- out[[1]]
    FPRout <- out[[2]]
    FPR_collect <- c(FPR_collect, FPRout)
    FNR_collect <- c(FNR_collect, FNRout)
}
    

plot(positive_WeightIn_list, FPR_collect, type="o", ylim=range(0:1), xlab = "Positive Weight", ylab="Metric", main = "XGBoost Adjusting the +ve Weights\n Cut-off: 0.04")
lines(positive_WeightIn_list, FNR_collect, type="o", col="red")


legend(0.8, 0.8, legend=c("FPR", "FNR"),
       col=c("black", "red"), lty=1:1, cex=0.8,  box.lty=0)


abline(h = 0.05, col="blue") 

# visualize: fine tune the grid 
for (i in seq(0,10,1)){
    i2 <- i / 10
    abline(h = i2, v = i, col="lightgray", lty="dotted") 
}

# visualize: fine tune the grid 
#for (i in seq(0,1000,100)){
#    i2 <- i / 1000
#    abline(h = i2, v = i, col="lightgray", lty="dotted") 
#}


results <- cbind(FNR_collect, FPR_collect, positive_WeightIn_list)
roundedData <- round_df(results, 3)
colnames(roundedData) <- c("FNR", "FPR", "Majority Class Weight")
roundedData<- as.data.frame(roundedData)
outty <- subset(roundedData, FNR < 0.05 & FPR < 0.5)
outty









##########################
## FEATURE IMPORTANCE ####
##########################

feat_data <- test1
keep <- names(feat_data[, -which(names(feat_data) %in% c("Cancer"))])
Xx <- as.matrix(feat_data[keep])
numeric_Xx <- apply(Xx, 2, as.numeric)
Yy <- as.matrix(feat_data$Cancer)
numeric_Yy <- as.numeric(Yy)
feat_model <- xgboost(data = numeric_Xx, label = numeric_Yy,
                      nrounds = 10, max_depth=20, objective = "binary:logistic", 
                      scale_pos_weight = 0.5, nthread = 4)  # make scale_pos_weight --> small = all 0's, increase it to a 1000, = more 1's #sumwneg / sumwpos.  #max_depth = 2 #set_weight(weight) 
importance_matrix <- xgb.importance(colnames(feat_data), model = feat_model)
importance_matrix 

xgb.plot.importance(importance_matrix, rel_to_first = TRUE, xlab = "Relative importance")
library(Ckmeans.1d.dp)

gg <- xgb.ggplot.importance(importance_matrix, measure = "Gain", rel_to_first = TRUE,  top_n = 10)
gg + ggplot2::ylab("Gain") + ggtitle("Feature Gain")

gg <- xgb.ggplot.importance(importance_matrix, measure = "Cover", rel_to_first = TRUE, top_n = 10)
gg + ggplot2::ylab("Cover") + ggtitle("Feature Cover")

gg <- xgb.ggplot.importance(importance_matrix, measure = "Frequency", rel_to_first = TRUE, top_n = 10)
gg + ggplot2::ylab("Frequency") + ggtitle("Feature Frequency")

gg <- xgb.ggplot.importance(importance_matrix, measure = "Importance", rel_to_first = TRUE, top_n = 10)
gg + ggplot2::ylab("Importance") + ggtitle("Top 10 Features")






### important for interpretability
Frame <- as.data.frame(cbind(importance_matrix$Feature, importance_matrix$Gain, importance_matrix$Cover, importance_matrix$Frequency, importance_matrix$Importance))
colnames(Frame) <- c("Feature", "Gain", "Cover", "Frequency", "Importance")
head(Frame)
