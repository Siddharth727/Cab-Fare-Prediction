#Removing all objects stored in R
rm(list=ls())

#Setting up current work directory
setwd("S:/Data science/Cab Project/In R")

#Checking the work directory
getwd()

#Loading Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees', "usdm")

#installing all packages in x
lapply(x, require, character.only = TRUE)
rm(x)

#Reading the train data csv file
traindata = read.csv("train_cab.csv", header = T)

#Understanding the structure of dataset
str(traindata)

#Changing the fare amount column from factor to numeric
traindata$fare_amount = as.numeric(as.character(traindata$fare_amount))

#Understanding the range of data
summary(traindata)

######################################## Outlier Analysis ###############################################

#Fare of cab cannot be negative or zero
traindata$fare_amount[traindata$fare_amount <= 0] = NA

#Replacing zeroes in latitude and longitudes with NA
traindata$pickup_longitude[traindata$pickup_longitude == 0] = NA
traindata$pickup_latitude[traindata$pickup_latitude == 0] = NA
traindata$dropoff_longitude[traindata$dropoff_longitude == 0] = NA
traindata$dropoff_latitude[traindata$dropoff_latitude == 0] = NA

#Removing passenger count above 6 and below 1
traindata=traindata[-which(traindata$passenger_count > 6),]
traindata=traindata[-which(traindata$passenger_count < 1),]

#Storing column names in cnames
cnames = colnames(traindata[,1:6])

#Loop to remove outliers from first six columns
for(i in cnames){
  print(i)
  val = traindata[,i][traindata[,i] %in% boxplot.stats(traindata[,i])$out]
  print(length(val))
  traindata = traindata[which(!traindata[,i] %in% val),]
}



###################################### Missing Value Analysis ###########################################

#Making a different dataframe to store the count of missing values in the train dataset columns
missing_val = data.frame(apply(traindata,2,function(x){sum(is.na(x))}))

#Making a new column with rown names of train data
missing_val$Columns = row.names(missing_val)

#Changing the name of the missing values column
names(missing_val)[1] =  "Missing_percentage"

#Finding the percentage of missing values in each column
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(traindata)) * 100

#Ordering them in decending order
missing_val = missing_val[order(-missing_val$Missing_percentage),]

#Emptying the row names
row.names(missing_val) = NULL

#Changing the columns order
missing_val = missing_val[,c(2,1)]

#Making a new csv file for missing value percentage
write.csv(missing_val,"MissingValuePercentage.csv", row.names = F)

missdata = traindata

#Taking a value in a dataset
#Actual Value = 5.30
#Mean Value = 8.57
#Median Value = 7.7
#KNN Value = 12.35
#We will take median for missing values in this dataset

missdata$fare_amount[is.na(missdata$fare_amount)] = median(missdata$fare_amount, na.rm = T)
missdata$pickup_latitude[is.na(missdata$pickup_latitude)] = median(missdata$pickup_latitude, na.rm = T)
missdata$pickup_longitude[is.na(missdata$pickup_longitude)] = median(missdata$pickup_longitude, na.rm = T)
missdata$dropoff_longitude[is.na(missdata$dropoff_longitude)] = median(missdata$dropoff_longitude, na.rm = T)
missdata$dropoff_latitude[is.na(missdata$dropoff_latitude)] = median(missdata$dropoff_latitude, na.rm = T)
missdata$passenger_count[is.na(missdata$passenger_count)] = median(missdata$passenger_count, na.rm = T)

#Putting data back to our main dataset
traindata = missdata

#Checking the missing values in dataset
sum(is.na(traindata))

################################################# Feature Selection ####################################################

#We are using correlation plot as the data is numeric
numeric_index = sapply(traindata,is.numeric)
corrgram(traindata[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

#In this data, we need both latitude and longitude to get the pickup and dropoff location although the correlation plot shows correlation between them
#Plus we are removing date and time feature as it is not much contributing to the model developmet

traindata = traindata[-c(2)]

#hist(traindata$fare_amount)
#hist(traindata$pickup_longitude)
#hist(traindata$pickup_latitude)
#hist(traindata$dropoff_longitude)
#hist(traindata$dropoff_latitude)
#hist(traindata$passenger_count)
#qqnorm(traindata$fare_amount)
#qqnorm(traindata$pickup_longitude)
#qqnorm(traindata$pickup_latitude)
#qqnorm(traindata$dropoff_longitude)
#qqnorm(traindata$dropoff_latitude)
#qqnorm(traindata$passenger_count)

############################################ Feature Scaling ###############################################
cnames = c("fare_amount", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count")

#for(i in cnames){
#  print(i)
#  traindata[,i] = (traindata[,i] - min(traindata[,i]))/
#    (max(traindata[,i] - min(traindata[,i])))
#}

#for(i in cnames){
#  print(i)
#  traindata[,i] = (traindata[,i] - mean(traindata[,i]))/
#    sd(traindata[,i])
#}

# We can do feature scaling by both normalization and standardization but the results are further downgrading that's why we are avoiding it

################################################ Model Development ##############################################

#Checking the multicollinearity
vif(traindata[,-1])
vifcor(traindata[,-1], th = 0.9)

#Using cross validation
train_in = sample(1:nrow(traindata), 0.8 * nrow(traindata))
train_0 = traindata[train_in,]
test_0 = traindata[-train_in,]


########################################## Linear Regression ##########################################################

model_lm1 = lm(fare_amount ~., data = train_0)

#Summary of the model
summary(model_lm1)

#Prediction
LM_predict = predict(model_lm1, test_0[,2:6])

MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

#Calculating MAPE
MAPE(test_0[,1], LM_predict)

#MAPE = 0.4118131
#Accuracy = 58.8 percent

#Plotting the graph 
qplot(x = test_0[,1], y = LM_predict, data = test_0, color = I("red"), geom = "point", xlab = "Test Data", ylab = "Predictions")

############################################# Decision Tree #######################################################

treefit = rpart(fare_amount ~ . ,data = train_0, method = "anova")

summary(treefit)

predictions_tt = predict(treefit, test_0[,2:6])

MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

MAPE(test_0[,1],predictions_tt)

#MAPE = 0.3268623
#Accuracy = 67.31 percent

qplot(x = test_0[,1], y = predictions_tt, data = test_0, color = I("red"), geom = "point", xlab = "Test Data", ylab = "Predictions")

############################################# Random Forest #######################################################

forestmodel = randomForest(fare_amount ~.,data=train_0)

summary(forestmodel)

forest_predictions = predict(forestmodel,test_0[,2:6])

MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

MAPE(test_0[,1],forest_predictions)

#MAPE = 0.2784401
#Accuracy = 72.15 percent

qplot(x = test_0[,1], y = forest_predictions, data = test_0, color = I("red"), geom = "point", xlab = "Test Data", ylab = "Predictions")

write.csv(forest_predictions, "predictions.csv", row.names = F)
