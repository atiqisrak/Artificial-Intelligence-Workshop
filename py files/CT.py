# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:21:01 2019

@author: rifaz
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import matplotlib.pyplot as plt 


df = pd.read_csv('C:\\Users\\rifaz\\Desktop\\py files\\iris.csv')
print(df.tail(5))

to_drop = ['virginica']
df.drop(to_drop, inplace=True, axis=1)

df.head()

df.shape
df.describe

df.plot(x='setosa', y='versicolor', style='o')  
plt.title('setosa vs versicolor')  
plt.xlabel('setosa')  
plt.ylabel('versicolor')  
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(df['versicolor'])

X = df['setosa'].values.reshape(-1,1)
y = df['versicolor'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

print(regressor.intercept_)

print(regressor.coef_)

y_pred = regressor.predict(X_test)

dx = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
dx

df1 = dx.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))