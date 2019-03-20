#MULTIPLE REGRESSION  with backward elimination 


#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
print(dataset)

#importing dataset, creation of matrix of features(explanatory variables)
#and the dependent variable vector
X = dataset.iloc[: , :-1].values
Y = dataset.iloc[: , 4].values

#encoding categorical data, 
#LabelEncoder for converting categories into numerical values
#OneHotEncoder for eliminating the hierarchical relation 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X= X[:,1:]


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#define variables at the same time
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2,
													random_state = 0)
													 


#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #object and the class
regressor.fit(X_train, Y_train)

#predicting the results of test_Set
y_pred = regressor.predict(X_test)
 


#Building the optimal model using the Backward Elimination
#first import library necessary for backward elimination
#we need to create a matrix of 50 lines of ones for the intercept 
import statsmodels.formula.api as sm
X= np.append(arr= np.ones((50,1)).astype(int), values =X , axis =1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS =sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#backward elimination -> we look for explanatory variables 
#statistically significant considering their p-value, threshold at alfa = .05
#removing the independent variable column index 2 (x2 has the highest p-value meaning )
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS =sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#still removing this time x1, 
X_opt = X[:,[0,3,4,5]]
regressor_OLS =sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#still removing, always look at original X 
X_opt = X[:,[0,3,5]]
regressor_OLS =sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#still removing not statistically significant explanatory variables 
X_opt = X[:,[0,3]]
regressor_OLS =sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

