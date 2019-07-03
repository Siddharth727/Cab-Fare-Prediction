#Importing Libraries
import os
import pandas as pd
import numpy as np
from fancyimpute import KNN  
import matplotlib.pyplot as plt
import seaborn as sns
from random import randrange, uniform
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#Setting work environment
os.chdir("S:/Data science/Cab Project/In Python")
os.getcwd()

#Loading train data
traindata = pd.read_csv("train_cab.csv", sep =",")

traindata.shape

traindata["fare_amount"].describe

#Changing the data type of fare_amount
traindata['fare_amount'] = pd.to_numeric(traindata['fare_amount'].str.replace('-' , ''))

traindata.info()

####################### Outlier Analysis ##################################

traincopy = traindata.copy()

sns.boxplot(x=traincopy['fare_amount'])

#Finding the upper and lower limits of fare_amount
traincopy.fare_amount.quantile([0.01,0.90])

#Doing data preprocessing after understanding the data
traincopy['fare_amount'][traincopy['fare_amount']<=0] = np.nan
traincopy['fare_amount'][traincopy['fare_amount']>20.5] = np.nan
traincopy['pickup_longitude'][traincopy['pickup_longitude']<=-180] = np.nan
traincopy['pickup_longitude'][traincopy['pickup_longitude']>=180] = np.nan
traincopy['pickup_latitude'][traincopy['pickup_latitude']<=-90] = np.nan
traincopy['pickup_latitude'][traincopy['pickup_latitude']>=90] = np.nan
traincopy['dropoff_longitude'][traincopy['dropoff_longitude']<=-180] = np.nan
traincopy['dropoff_longitude'][traincopy['dropoff_longitude']>=180] = np.nan
traincopy['dropoff_latitude'][traincopy['dropoff_latitude']<=-90] = np.nan
traincopy['dropoff_latitude'][traincopy['dropoff_latitude']>=90] = np.nan

traincopy['passenger_count'][traincopy['passenger_count']<1] = np.nan
traincopy['passenger_count'][traincopy['passenger_count']>6] = np.nan
cdist = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']
for i in cdist:
    traincopy[i][traincopy[i]==0] = np.nan


######################## Missing Value Analysis #######################################

#Create dataframe with missing percentage
mvalues = pd.DataFrame(traincopy.isnull().sum())

#Reset index
mvalues = mvalues.reset_index()

#Rename variable
mvalues = mvalues.rename(columns = {'index': 'Variables', 0: 'Percentages'})

#Calculate percentage
mvalues['Percentages'] = (mvalues['Percentages']/len(traincopy))*100

#descending order
mvalues = mvalues.sort_values('Percentages', ascending = False).reset_index(drop = True)

#save output results 
mvalues.to_csv("Misspercentages.csv", index = False)

#Value = 5.3
#Mean = 13.3406
#Median = 7.7
#KNN = 12.35
#Creating missing value
tf=traincopy.copy()
tf['fare_amount'].loc[36] = np.nan

#Impute with mean
#tf['fare_amount'] = tf['fare_amount'].fillna(tf['fare_amount'].mean())

#Impute with median
#tf['fare_amount'] = tf['fare_amount'].fillna(tf['fare_amount'].median())

#Using KNN method
#tf = pd.DataFrame(KNN(k = 5).complete(tf), columns = tf.columns)

cnm = ['fare_amount', 'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']
for i in cnm:
    tf[i] = tf[i].fillna(tf[i].median())

traincopy = tf.copy()
########################### Feature Selection ########################################
#We are dropping pickup_datetime column as it is of no relevance
traincopy = traincopy.drop(['pickup_datetime'], axis=1)

########################### Feature Scaling ###########################################
#Result same or worse with feature scaling
#for i in cnm:
    #traincopy[i] = (traincopy[i] - min(traincopy[i]))/(max(traincopy[i]) - min(traincopy[i]))
    #traincopy[i] = (traincopy[i] - traincopy[i].mean())/traincopy[i].std()
########################## Model Development ##########################################
#Using cross validation for traion and test split
train, test = train_test_split(traincopy, test_size=0.2)

#Linear regression using ordinary least square method
linearmodel = sm.OLS(train.iloc[:,0], train.iloc[:,1:6]).fit()

#Summary of linear regression model
linearmodel.summary()

#Predicting through linear regression model
linearpredictions = linearmodel.predict(test.iloc[:,1:6])

#Using MAPE to find error
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape
 
MAPE(test.iloc[:,0], linearpredictions)

#Error = 37.677871884016334
#Accuracy = 62.33

#Decision tree for regression
fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,1:6], train.iloc[:,0])

#Applying model on test data
predictions_DT = fit_DT.predict(test.iloc[:,1:6])

#Using MAPE to find error
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape

MAPE(test.iloc[:,0], predictions_DT)

#Error = 37.160747137558836
#Accuracy = 62.84

#Random Forest for regression
randomf = RandomForestRegressor(n_estimators = 1000)

#Fitting the model
randomf.fit(train.iloc[:,1:6], train.iloc[:,0])

predictions_RF = randomf.predict(test.iloc[:,1:6])

#Using MAPE to find error
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape

MAPE(test.iloc[:,0], predictions_RF)

#Error = 21.38598493937194
#Accuracy = 78.62

#Plotting the output
sns.scatterplot(x=test["fare_amount"], y=predictions_RF)
