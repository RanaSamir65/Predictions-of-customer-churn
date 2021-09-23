#Reading the dataset
data = read.csv("BankChurners.csv", stringsAsFactors = FALSE)
str(data)
summary(data)



#Preparing the dataset
data$Attrition_Flag = ifelse(data$Attrition_Flag == "Existing Customer", 0, 1)
data$CLIENTNUM = NULL
data$Gender = ifelse(data$Gender == "F", 1, 0)
data$Education_Level = ifelse(data$Education_Level == "Unknown", 1, ifelse(data$Education_Level == "Uneducated", 2, ifelse(data$Education_Level == "High School", 3, ifelse(data$Education_Level == "College", 4, ifelse(data$Education_Level == "Graduate", 5, ifelse(data$Education_Level == "Post-Graduate",6,ifelse(data$Education_Level == "Doctorate",7,8)))))))
data$Income_Category = ifelse(data$Income_Category == "Unknown", 1, ifelse(data$Income_Category == "Less than $40K", 2, ifelse(data$Income_Category == "$40K - $60K", 3, ifelse(data$Income_Category == "$60K - $80K", 4, ifelse(data$Income_Category == "$80K - $120K", 5, ifelse(data$Income_Category == "$120K +",6,7))))))
data$Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1 = NULL
data$Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2 = NULL

#Decomposing the marital status variable into new variables
Marital_Status = unique(data$Marital_Status)
Marital_Status
for (a in Marital_Status){
  assign(paste(a), ifelse(data$Marital_Status == a,1,0))
}
data$Single = Single
data$Married = Married
data$Marital_Status_Unknown = Unknown
data$Divorced= Divorced
data$Marital_Status = NULL


#Decomposing the card category variable into new variables
Card_Category = unique(data$Card_Category)
Card_Category
for (c in Card_Category){
  assign(paste(c), ifelse(data$Card_Category == c,1,0))
}
data$Blue = Blue
data$Gold = Gold
data$Platinum = Platinum
data$Silver= Silver
data$Card_Category = NULL


#splitting the data set into training and testing set
library(caTools)
set.seed(144)
spl = sample.split(data$Attrition_Flag, SplitRatio=0.7)
TrainData = subset(data, spl==TRUE)
TestData = subset(data, spl==FALSE)




#Baseline model accuracy
table(TestData$Attrition_Flag)
#2550/3038 
#0.839368

#Using Logistic Regression model
LogReg = glm(Attrition_Flag~. ,  data = TrainData, family = "binomial")
summary(LogReg)
#AIC: 3333.1


#removing Marital_Status_Unknown
TrainData$Marital_Status_Unknown=NULL
#AIC: 3333.1

#removing Education level
TrainData$Education_Level=NULL
#AIC:  3331.1

#removing Single
TrainData$Single=NULL
#AIC:  3329.3


#removing Customer Age
TrainData$Customer_Age=NULL
#AIC:  3327.6


#removing Platinum
TrainData$Platinum=NULL
#AIC:  3333.1



#removing Divorced
TrainData$Divorced= NULL
#AIC:  3331.7


#removing Gold
TrainData$Gold= NULL
#AIC:  3333.9


#removing Credit Limit 
TrainData$Credit_Limit= NULL
#AIC:  3333.9


#removing Avg_Utilization_Ratio 
TrainData$Avg_Utilization_Ratio= NULL
#AIC:  3331.1





#The model after removing all of the above insignificant variables
LogRegS = glm(Attrition_Flag~. ,  data = TrainData, family = "binomial")
summary(LogRegS)
# AIC: 3321.7



#prediction using logistic regression model
PredLogReg= predict(LogRegS,newdata= TestData, type = "response")




#ROC Curve 
library(ROCR)
ROCRpred = prediction(PredLogReg, TestData$Attrition_Flag)
ROCRperf = performance(ROCRpred, "tpr", "fpr")
plot(ROCRperf, colorize = TRUE,print.cutoffs.at=seq(0,1,by=0.1),text.adj=c(-0.5,0.5))

# Logistic Regression model when Prediction  using a threshold of 0.1
table(TestData$Attrition_Flag, PredLogReg>0.1)
Acc = (1983+430) / 3038
Acc
#0.7942725 


# Logistic Regression model Prediction when using a threshold of 0.2
table(TestData$Attrition_Flag, PredLogReg>0.2)
Acc = (2248+394) / 3038
Acc
#0.8696511


# Logistic Regression model Prediction when using a threshold of 0.3
table(TestData$Attrition_Flag, PredLogReg>0.3)
Accuracy = (2355+351) / 3038
Accuracy
#0.8907176
senstivity = 351/(351+137) 
senstivity
#0.7192623

specificity = 2355/(2355+195)
specificity
#0.9235294





#Using CART model
library(rpart)
library(rpart.plot)
CART = rpart(Attrition_Flag~., data = TrainData, method = "class")
summary(CART)
prp(CART)
PredCART = predict(CART, newdata= TestData, type = "class")


#CART model after changing the cp

#determine the cp to be used
library(caret)
library(lattice)
library(ggplot2)
set.seed(2)
Cpcontrol = trainControl( method = "cv", number = 10 )
Grid = data.frame(.cp = seq(0.0,0.092,0.001))
cartCV = train(Attrition_Flag~. , data = TrainData, method = "rpart", trControl = Cpcontrol, tuneGrid = Grid )
cartCV
plot(cartCV)


#CART model using the cp
CARTWithCP = rpart(Attrition_Flag~., data = TrainData, method = "class", cp = 0.002)
prp(CARTWithCP)
PredCARTWithcp = predict(CARTWithCP, newdata= TestData)


#ROC Curve 
library(ROCR)
PredCARTROC = predict(CARTWithCP, newdata= TestData)
pred.prob = PredCARTROC[,2]

ROCRCARTpred = prediction(pred.prob, TestData$Attrition_Flag)
ROCRCARTperf = performance(ROCRCARTpred,"tpr","fpr")
plot(ROCRCARTperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1),text.adj=c(-0.5,0.5))


# CART model Prediction when using a threshold of 0.1
table(TestData$Attrition_Flag, pred.prob>0.1)
Acc = (406+2423)/3038
Acc
#0.9312047


# CART model Prediction when using a threshold of 0.2
table(TestData$Attrition_Flag, pred.prob>0.2)
Acc = (388+2456)/3038
Acc
#0.9361422

# CART model Prediction when using a threshold of 0.3
table(TestData$Attrition_Flag, pred.prob>0.3)
Acc = (372+2483)/3038
Acc
#0.939763


# CART model Prediction when using a threshold of 0.4
table(TestData$Attrition_Flag, pred.prob>0.4)
Acc = (372+2483)/3038
Acc
#0.939763


# CART model Prediction when using a threshold of 0.5
table(TestData$Attrition_Flag, pred.prob>0.5)
Acc = (372+2483)/3038
Acc
#0.939763


# CART model Prediction when using a threshold of 0.6
table(TestData$Attrition_Flag, pred.prob>0.6)
Acc = (371+2489)/3038
Acc
#0.9414088


sensitivity = (371)/(371+117)
sensitivity
#0.7602459

specificity = (2489)/(2489+61)
specificity
#0.9760784


# CART model Prediction when using a threshold of 0.7
table(TestData$Attrition_Flag, pred.prob>0.7)
Acc = (365+2495)/3038
Acc
#0.9414088



# CART model Prediction when using a threshold of 0.8
table(TestData$Attrition_Flag, pred.prob>0.8)
Acc = (318+2507)/3038
Acc
#0.9298881







#Use random forest model
library(randomForest)
set.seed(1)
RanFor = randomForest(Attrition_Flag ~ . , data = TrainData, importance= TRUE)
RanFor

#plotting trees vs error
plot(RanFor, type = "simple")

#Random Forest Model Error_Rate using ntree of 200
RanFor2 = randomForest(Attrition_Flag ~ . , data = TrainData,ntree = 200, importance= TRUE)
RanFor2
#Mean of squared residuals: 0.0302218


#Random Forest Model Error_Rate using ntree of 300
RanFor3 = randomForest(Attrition_Flag ~ . , data = TrainData,ntree = 300, importance= TRUE)
RanFor3
#Mean of squared residuals: 0.03035005


#Random Forest Model Error_Rate using ntree of 400
RanFor4 = randomForest(Attrition_Flag ~ . , data = TrainData,ntree = 400, importance= TRUE)
RanFor4
#Mean of squared residuals: 0.03030569


#Random Forest Model Error_Rate using ntree of 500
RanFor5 = randomForest(Attrition_Flag ~ . , data = TrainData,ntree = 500, importance= TRUE)
RanFor5
#Mean of squared residuals: 0.03031549


#plotting the significant variables for the random forest model
varImpPlot(RanFor5, sort=TRUE, n.var=(15))



#ROC Curve 
library(ROCR)
PredRanFor = predict(RanFor5, newdata=TestData)
ROCRRandForpred = prediction(PredRanFor, TestData$Attrition_Flag)
ROCRRandperf = performance(ROCRRandForpred, "tpr", "fpr")
plot(ROCRRandperf, colorize = TRUE,print.cutoffs.at=seq(0,1,by=0.1),text.adj=c(-0.5,0.5))


#Random Forest model Prediction when using a threshold of 0.1
table(TestData$Attrition_Flag, PredRanFor>0.1)
AccuracyRand = (479+2192) / 3038
AccuracyRand

#0.8791968


#Random Forest model Prediction when using a threshold of 0.2
table(TestData$Attrition_Flag, PredRanFor>0.2)
AccuracyRand = (467+2372) / 3038
AccuracyRand

#0.9344964


#Random Forest model Prediction when using a threshold of 0.3
table(TestData$Attrition_Flag, PredRanFor>0.3)
AccuracyRand = (454+2448) / 3038
AccuracyRand

#0.9552337



#Random Forest model Prediction when using a threshold of 0.4
table(TestData$Attrition_Flag, PredRanFor>0.4)
AccuracyRand = (435+2497) / 3038
AccuracyRand

#0.9651086


#Random Forest model Prediction when using a threshold of 0.5
table(TestData$Attrition_Flag, PredRanFor>0.5)
AccuracyRand = (413+2528) / 3038
AccuracyRand

#0.9680711

sensitivity = (413)/(413+75)
sensitivity
#0.8463115

specificity = (2528)/(2528+22)
specificity
#0.9913725


#Random Forest model Prediction when using a threshold of 0.6
table(TestData$Attrition_Flag, PredRanFor>0.6)
AccuracyRand = (374+2539) / 3038
AccuracyRand

#0.9588545










