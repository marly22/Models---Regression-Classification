#Logistic regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')


#importing dataset, we want to predict response yes/no
#based on age and salary - explanatory variables
X = dataset.iloc[: , 2:4].values
Y = dataset.iloc[: , 4].values


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#we have 400 observations so we can do a split 300 and 100
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, 
random_state = 0)

#feature scaling, we have x variables in range 0 and 1 , 
#easier for computational things
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Logistic Regression to training set 
#importing library - linear model libray
#logistic regression is a linear classifier
#2 categories (response 0 and 1) will be separated by straight line
#linear_model is the name of library, LogisticRegression is a class
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, Y_train) #classifier object fitted into training set, so it learns correlation 
#classifier's predictive power will be tested on test set

#predictig test set results, y_pred -> vector of prediction
y_pred = classifier.predict(X_test)

#making the Confusion Matrix 
#(contain the correct predictions that our model made on test set as well as incorrect predictions)
#we import FUNCTION from matrix library
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
#cm in console shows an array 65,24 are correct predictions and 3,8are incorrect
#cm important to evaluate the power of our prediction model

#Visualizing the training set results
# WE SEE true results and regions od predictions
#observations points of training set : red points users who didn't buy and green one which bought
#we can deduce that users: young and with low salary didn't buy SUV car
#the goal of classfication is to classify the right users to right categories
#we plot 2 prediction regions: red for non-buyers and green for buyers( logistic classifier)
#we have straight line(straight because our logistic classifier is linear, which is not for non-linear classifiers)
#  between 2 regions: PREDICTION BOUNDARY 
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#we took all pixel points and and we applied our classifier on it; each of this pixel point
#is a user of socialnetwork with its own salary and age
#so if a classifier predicts 0 it will colorize pixel in red and if predicts 1 , it will colorize in green
#so with meshgrid we prepare all pixel points; we take min value of age -1 cuz we don't want points to be 
#squeezed on axes
#we plot the contour, we apply classifier to predict if it's red or green 
#with loop we plotted all real valyes 

#see how our logistic classifier will predict response for new observations
#based on its learning experience on training set

#visualisizing Test set results
#predicted quite well, few mistakes
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()