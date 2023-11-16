#!/usr/bin/env python
# coding: utf-8

# # Group members:
# 
# 1. Ashok Lamichhane (21201785)
# 
# 2. Mukund Prasad Singh (20201202)
# 
# 3. Udoy Saha (23341134)
# 
# 4. Noor E Jannat Nafia (21301631)

# **We will predict whether it will be beneficial to purchase a stock of a company on a particular day.** <br>
# (A stock purchase is only benificial when the closing price of the day is higher than the opening price of a day.)<br><br>
# 
# 
# Dataset source - Kaggle<br>
# Link - https://www.kaggle.com/datasets/khushipitroda/stock-market-historical-data-of-top-10-companies<br>

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# all libraries as needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# # **Reading dataset and inserting into our dataframe**
# 

# In[ ]:


#dataframe = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/CSE422/data.csv")
dataframe = pd.read_csv("/content/drive/My Drive/CSE422 Lab/CSE422 Project Group 1/data.csv")
dataframe.head()


# In[ ]:


# Finding out the number of companies present
companies = dataframe['Company'].unique()
print("All companies:")
for company in companies:
  print(company)


# In[ ]:


# As we will predict the opening and closing price for a specific company, choose a company
predicting_company = input("Enter the company name to predict from the list above : ").upper()
print("Company to predict : " + predicting_company)


# In[ ]:


# Selecting records for the chosen company
dataframe = dataframe[dataframe['Company'] == predicting_company]
dataframe


# In[ ]:


# Information about the columns
dataframe.info()


# There is some issues in this dataset. They are: <br>
# * The values of Date column needs to be formatted to Pandas 'datetime' format
# * Columns Close/Last, Open, High and Low needs to be converted into numerical type
# * Some Null values need to be Imputed from columns:
#   *   Close/Last
#   *   Volume
#   *   Open
#   *   High
#   *   Low

# In[ ]:


# Converting Date column into Timestamp format
dataframe['Date'] = pd.to_datetime(dataframe['Date'])
dataframe.head()


# In[ ]:


# Timestamp format is not appropriate for training. So, it is converted into the count of day from the initial day.
initial_day = dataframe['Date'].min()
# Total = (dataframe.Date.max() - initial_day).days


# In[ ]:


# Converting columns Date, Close/Last, Open, High and Low to appropriate type
dataframe["Date"] = dataframe["Date"].apply(lambda row: (row-initial_day).days)
dataframe["Close/Last"] = dataframe["Close/Last"].apply(lambda row: float(row[1:]) if type(row) == str else row)
dataframe["Open"] = dataframe["Open"].apply(lambda row: float(row[1:]) if type(row) == str else row)
dataframe["High"] = dataframe["High"].apply(lambda row: float(row[1:]) if type(row) == str else row)
dataframe["Low"] = dataframe["Low"].apply(lambda row: float(row[1:]) if type(row) == str else row)
dataframe.head()


# In[ ]:


# Data imputation
impute = SimpleImputer(missing_values=np.nan, strategy='mean')

impute.fit(dataframe[['Close/Last', 'Volume', 'Open', 'High', 'Low']])

dataframe[['Close/Last', 'Volume', 'Open', 'High', 'Low']] = impute.transform(dataframe[['Close/Last', 'Volume', 'Open', 'High', 'Low']])


# In[ ]:


dataframe.head()


# **By now, the dataframe does not have any null values.<br>
# The datatypes are also adjusted.**
# 
# # Exploratory Data Analysis

# In[ ]:


# Information about the columns
dataframe.info()


# In[ ]:


# Description of the column values
dataframe.describe()


# In[ ]:


# Correlation heatmap
sns.heatmap(dataframe.corr(), cmap = 'YlGnBu')


# # The Opening and Closing price will be predicted. High ad Low price is highly correlated with those. Since those columns add no extra information, those are dropped.

# In[ ]:


dataframe = dataframe.drop('High', axis=1)
dataframe = dataframe.drop('Low', axis=1)
dataframe


# In[ ]:


# Visualizing the Data
sns.lineplot(x = "Date", y = "Open", data = dataframe, color = "magenta", label = 'Open', linewidth = 0.5)
sns.lineplot(x = "Date", y = "Close/Last", data = dataframe, color = "blue", label = 'Close', linewidth = 0.5)
plt.xlabel('Day')
plt.ylabel('Open and Close/Last Price')


# # All features and the output variables are separated at this point
# 

# In[ ]:


X = dataframe[['Date', 'Volume']]
# X = dataframe[['Date']]
X


# In[ ]:


Y_Open = dataframe['Open']
Y_Open


# In[ ]:


Y_Close = dataframe['Close/Last']
Y_Close


# # Train Test Split section

# In[ ]:


xTrain, xTest, yOpenTrain, yOpenTest = train_test_split(X, Y_Open, test_size = .3, random_state = 1)
xTrain, xTest, yCloseTrain, yCloseTest = train_test_split(X, Y_Close, test_size = .3, random_state = 1)


# In[ ]:


xTrain


# In[ ]:


xTest


# In[ ]:


yOpenTrain


# In[ ]:


yOpenTest


# In[ ]:


yCloseTrain


# In[ ]:


yCloseTest


# # In this step, feature scaling is performed

# In[ ]:


# This is for MinMaxScaler scaling technique
scaler1 = MinMaxScaler()
scaler1.fit(xTrain)

xTrain_scaled1 = scaler1.transform(xTrain)
xTest_scaled1 = scaler1.transform(xTest)

# Fit with xTrain_scaled1 & yOpenTrain/yCloseTrain
# Predict with xTest_scaled1
# Score with xTest_scaled1 & yOpenTest/yCloseTest


# In[ ]:


xTrain_scaled1


# In[ ]:


xTest_scaled1


# In[ ]:


# This is for StandardScaler scaling technique
scaler2 = StandardScaler()
scaler2.fit(xTrain)

xTrain_scaled2 = scaler2.transform(xTrain)
xTest_scaled2 = scaler2.transform(xTest)

# Fit with xTrain_scaled2 & yOpenTrain/yCloseTrain
# Predict with xTest_scaled2
# Score with xTest_scaled2 & yOpenTest/yCloseTest


# In[ ]:


xTrain_scaled2


# In[ ]:


xTest_scaled2


# # Fitting into models
# **Now different models can be fitted using processed data**

# In[ ]:


# Finally, a function to test the score of predicting Profit

def profit_score(yClosePrediction, yOpenPrediction):
  predicted = yClosePrediction - yOpenPrediction
  actual = (yCloseTest - yOpenTest).to_numpy()

  for i in range(len(predicted)):
    if predicted[i] > 0:
      predicted[i] = True
    else:
      predicted[i] = False

  for i in range(len(actual)):
    if actual[i] > 0:
      actual[i] = True
    else:
      actual[i] = False

  matched_predictions = 0

  for i in range(len(predicted)):
    if predicted[i] == actual[i]:
      matched_predictions += 1

  print(matched_predictions*100/len(predicted), "%")


# 
# 
# # *   Linear regression
# 
# 
# 

# In[ ]:


from sklearn.linear_model import LinearRegression
linReg = LinearRegression()


# In[ ]:


# Firstly using scaler1

# For Open price
linReg.fit(xTrain_scaled1, yOpenTrain)

yOpenPrediction1 = linReg.predict(xTest_scaled1)


# In[ ]:


plt.scatter(xTest_scaled1[:, 0], yOpenTest, color = 'red', marker = '.')
plt.plot(xTest_scaled1[:, 0], linReg.predict(xTest_scaled1), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Open Price')
plt.show()


# In[ ]:


print(linReg.score(xTest_scaled1, yOpenTest)*100, "%")


# In[ ]:


# For Close price
linReg.fit(xTrain_scaled1, yCloseTrain)

yClosePrediction1 = linReg.predict(xTest_scaled1)


# In[ ]:


plt.scatter(xTest_scaled1[:, 0], yCloseTest, color = 'red', marker = '.')
plt.plot(xTest_scaled1[:, 0], linReg.predict(xTest_scaled1), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Close/Last Price')
plt.show()


# In[ ]:


print(linReg.score(xTest_scaled1, yCloseTest)*100, "%")


# In[ ]:


profit_score(yClosePrediction1, yOpenPrediction1)


# In[ ]:





# In[ ]:


# Secondly using scaler2

# For Open price
linReg.fit(xTrain_scaled2, yOpenTrain)

yOpenPrediction2 = linReg.predict(xTest_scaled2)


# In[ ]:


plt.scatter(xTest_scaled2[:, 0], yOpenTest, color = 'red', marker = '.')
plt.plot(xTest_scaled2[:, 0], linReg.predict(xTest_scaled2), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Open Price')
plt.show()


# In[ ]:


print(linReg.score(xTest_scaled2, yOpenTest)*100, "%")


# In[ ]:


# For Close price
linReg.fit(xTrain_scaled2, yCloseTrain)

yClosePrediction2 = linReg.predict(xTest_scaled2)


# In[ ]:


plt.scatter(xTest_scaled2[:, 0], yCloseTest, color = 'red', marker = '.')
plt.plot(xTest_scaled2[:, 0], linReg.predict(xTest_scaled2), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Close/Last Price')
plt.show()


# In[ ]:


print(linReg.score(xTest_scaled2, yCloseTest)*100, "%")


# In[ ]:


profit_score(yClosePrediction2, yOpenPrediction2)


# # **Support Vector machine**
# For Scale 1
# 

# In[ ]:


#For Open Price
from sklearn.svm import SVR

# Create support vector regression model and fit the data to it
model = SVR(kernel='rbf', C=1e3, gamma=0.1) # Radial basis function kernel
model.fit(xTrain_scaled1 , yOpenTrain)

# Create testing data points
yOpenPrediction3 = model.predict(xTest_scaled1)

plt.scatter(xTrain_scaled1[:, 0], yOpenTrain, color = 'red', marker = '.')
plt.plot(xTrain_scaled1[:, 0], model.predict(xTrain_scaled1), color = 'blue', linewidth = 0.1)
plt.title('Training set')
plt.xlabel('Days')
plt.ylabel('Open Price')
plt.show()


# In[ ]:


print(model.score(xTest_scaled1, yCloseTest)*100, "%")


# In[ ]:


#For Close Price

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


# Create support vector regression model and fit the data to it
model1 = SVR(kernel='rbf', C=1e3, gamma=0.1) # Radial basis function kernel
model1.fit(xTrain_scaled1 , yCloseTrain)

# Create testing data points
yClosePrediction3 = model1.predict(xTest_scaled1)

plt.scatter(xTrain_scaled1[:, 0], yCloseTrain, color = 'red', marker = '.')
plt.plot(xTrain_scaled1[:, 0], model.predict(xTrain_scaled1), color = 'blue', linewidth = 0.1)
plt.title('Training set')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.show()


# In[ ]:


print(model1.score(xTest_scaled1, yCloseTest)*100, "%")


# In[ ]:


profit_score(yClosePrediction3, yOpenPrediction3)


# ### For Scale 2
# 

# In[ ]:


#For Open Price

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


# Create support vector regression model and fit the data to it
model = SVR(kernel='rbf', C=1e3, gamma=0.1) # Radial basis function kernel
model.fit(xTrain_scaled2 , yOpenTrain)

# Create testing data points
yOpenPrediction4 = model.predict(xTest_scaled2)

plt.scatter(xTrain_scaled2[:, 0], yOpenTrain, color = 'red', marker = '.')
plt.plot(xTrain_scaled2[:, 0], model.predict(xTrain_scaled2), color = 'blue', linewidth = 0.1)
plt.title('Training set')
plt.xlabel('Days')
plt.ylabel('Open Price')
plt.show()


# In[ ]:


print(model.score(xTest_scaled2, yCloseTest)*100, "%")


# In[ ]:


#For Close Price

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


# Create support vector regression model and fit the data to it
model1 = SVR(kernel='rbf', C=1e3, gamma=0.1) # Radial basis function kernel
model1.fit(xTrain_scaled2 , yCloseTrain)

# Create testing data points
yClosePrediction4 = model1.predict(xTest_scaled2)

plt.scatter(xTrain_scaled2[:, 0], yCloseTrain, color = 'red', marker = '.')
plt.plot(xTrain_scaled2[:, 0], model1.predict(xTrain_scaled2), color = 'blue', linewidth = 0.1)
plt.title('Training set')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.show()


# In[ ]:


print(model1.score(xTest_scaled2, yCloseTest)*100, "%")


# In[ ]:


profit_score(yClosePrediction4, yOpenPrediction4)


# * # **Decision Tree**

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state = 0)


# In[ ]:


# Firstly using scaler1

# For Open price
dt.fit(xTrain_scaled1, yOpenTrain)

yOpenPrediction5 = dt.predict(xTest_scaled1)


# In[ ]:


plt.scatter(xTest_scaled1[:, 0], yOpenTest, color = 'red', marker = '.')
plt.plot(xTest_scaled1[:, 0], dt.predict(xTest_scaled1), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Open Price')
plt.show()


# In[ ]:


print(dt.score(xTest_scaled1, yOpenTest)*100, "%")


# In[ ]:


# For Close price
dt.fit(xTrain_scaled1, yCloseTrain)

yClosePrediction5 = dt.predict(xTest_scaled1)


# In[ ]:


plt.scatter(xTest_scaled1[:, 0], yCloseTest, color = 'red', marker = '.')
plt.plot(xTest_scaled1[:, 0], dt.predict(xTest_scaled1), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Close/Last Price')
plt.show()


# In[ ]:


print(dt.score(xTest_scaled1, yCloseTest)*100, "%")


# In[ ]:


profit_score(yClosePrediction5, yOpenPrediction5)


# In[ ]:





# In[ ]:


# Secondly using scaler2

# For Open price
dt.fit(xTrain_scaled2, yOpenTrain)

yOpenPrediction6 = dt.predict(xTest_scaled2)


# In[ ]:


plt.scatter(xTest_scaled2[:, 0], yOpenTest, color = 'red', marker = '.')
plt.plot(xTest_scaled2[:, 0], dt.predict(xTest_scaled2), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Open Price')
plt.show()


# In[ ]:


print(dt.score(xTest_scaled2, yOpenTest)*100, "%")


# In[ ]:


# For Close price
dt.fit(xTrain_scaled2, yCloseTrain)

yClosePrediction6 = dt.predict(xTest_scaled2)


# In[ ]:


plt.scatter(xTest_scaled2[:, 0], yCloseTest, color = 'red', marker = '.')
plt.plot(xTest_scaled2[:, 0], dt.predict(xTest_scaled2), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Close/Last Price')
plt.show()


# In[ ]:


print(dt.score(xTest_scaled2, yCloseTest)*100, "%")


# In[ ]:


profit_score(yClosePrediction6, yOpenPrediction6)


# # *Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=0)


# In[ ]:


# Firstly using scaler1

# For Open price
rf.fit(xTrain_scaled1, yOpenTrain)

yOpenPrediction7 = rf.predict(xTest_scaled1)


# In[ ]:


plt.scatter(xTest_scaled1[:, 0], yOpenTest, color = 'red', marker = '.')
plt.plot(xTest_scaled1[:, 0], rf.predict(xTest_scaled1), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Open Price')
plt.show()


# In[ ]:


print(rf.score(xTest_scaled1, yOpenTest)*100, "%")


# In[ ]:


# For Close price
rf.fit(xTrain_scaled1, yCloseTrain)

yClosePrediction7 = rf.predict(xTest_scaled1)


# In[ ]:


plt.scatter(xTest_scaled1[:, 0], yCloseTest, color = 'red', marker = '.')
plt.plot(xTest_scaled1[:, 0], rf.predict(xTest_scaled1), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Close/Last Price')
plt.show()


# In[ ]:


print(rf.score(xTest_scaled1, yCloseTest)*100, "%")


# In[ ]:


profit_score(yClosePrediction7, yOpenPrediction7)


# In[ ]:





# In[ ]:


# Secondly using scaler2

# For Open price
rf.fit(xTrain_scaled2, yOpenTrain)

yOpenPrediction8 = rf.predict(xTest_scaled2)


# In[ ]:


plt.scatter(xTest_scaled2[:, 0], yOpenTest, color = 'red', marker = '.')
plt.plot(xTest_scaled2[:, 0], rf.predict(xTest_scaled2), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Open Price')
plt.show()


# In[ ]:


print(rf.score(xTest_scaled2, yOpenTest)*100, "%")


# In[ ]:


# For Close price
rf.fit(xTrain_scaled2, yCloseTrain)

yClosePrediction8 = rf.predict(xTest_scaled2)


# In[ ]:


plt.scatter(xTest_scaled2[:, 0], yCloseTest, color = 'red', marker = '.')
plt.plot(xTest_scaled2[:, 0], rf.predict(xTest_scaled2), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Close/Last Price')
plt.show()


# In[ ]:


print(rf.score(xTest_scaled2, yCloseTest)*100, "%")


# In[ ]:


profit_score(yClosePrediction8, yOpenPrediction8)


# # * KNN
# 

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor(n_neighbors = 8)


# In[ ]:


# Firstly using scaler1

# For Open price
knnr.fit(xTrain_scaled1, yOpenTrain)

yOpenPrediction9 = knnr.predict(xTest_scaled1)


# In[ ]:


plt.scatter(xTest_scaled1[:, 0], yOpenTest, color = 'red', marker = '.')
plt.plot(xTest_scaled1[:, 0], knnr.predict(xTest_scaled1), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Open Price')
plt.show()


# In[ ]:


print(knnr.score(xTest_scaled1, yOpenTest)*100, "%")


# In[ ]:


# For Close price
knnr.fit(xTrain_scaled1, yCloseTrain)

yClosePrediction9 = knnr.predict(xTest_scaled1)


# In[ ]:


plt.scatter(xTest_scaled1[:, 0], yCloseTest, color = 'red', marker = '.')
plt.plot(xTest_scaled1[:, 0], knnr.predict(xTest_scaled1), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Close/Last Price')
plt.show()


# In[ ]:


print(knnr.score(xTest_scaled1, yCloseTest)*100, "%")


# In[ ]:


profit_score(yClosePrediction9, yOpenPrediction9)


# In[ ]:





# In[ ]:


# Secondly using scaler2

# For Open price
knnr.fit(xTrain_scaled2, yOpenTrain)

yOpenPrediction10 = knnr.predict(xTest_scaled2)


# In[ ]:


plt.scatter(xTest_scaled2[:, 0], yOpenTest, color = 'red', marker = '.')
plt.plot(xTest_scaled2[:, 0], knnr.predict(xTest_scaled2), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Open Price')
plt.show()


# In[ ]:


print(knnr.score(xTest_scaled2, yOpenTest)*100, "%")


# In[ ]:


# For Close price
knnr.fit(xTrain_scaled2, yCloseTrain)

yClosePrediction10 = knnr.predict(xTest_scaled2)


# In[ ]:


plt.scatter(xTest_scaled2[:, 0], yCloseTest, color = 'red', marker = '.')
plt.plot(xTest_scaled2[:, 0], knnr.predict(xTest_scaled2), color = 'blue', linewidth = 0.1)
plt.title('Test set')
plt.xlabel('Days')
plt.ylabel('Close/Last Price')
plt.show()


# In[ ]:


print(knnr.score(xTest_scaled2, yCloseTest)*100, "%")


# In[ ]:


profit_score(yClosePrediction10, yOpenPrediction10)


# # **Conclusion**<br>
# 
# In the linear regression model we got almost 83% accuracy whereas from the SVR model we got more accuracy 93%. When we use Decision Tree, Random Forest, KNN model we got accuracy 99.3%, 99.4% & 98.9% respectively. It seems among our five models Decision Tree and Random forest have better performance.<br>
# 
# We calculated the profit using our five different models by just calculating/using  the open price and close price(profit). The results seem like 50.33% for Linear Regression, 53.24% for SVR, 47.81% for Decision Tree, 49.60% for Random Forest, 45.96% for KNN. It seems SVR predicts better profit than other models.
