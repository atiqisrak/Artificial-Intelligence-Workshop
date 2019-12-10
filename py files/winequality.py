import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('C:\\Users\\rifaz\\Desktop\\py files\\winequality.csv')
print(dataset.head(5))


dataset.isnull().any()
dataset = dataset.fillna(method='ffill')

dataset.columns

X = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)

X.shape

#Spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
										test_size=0.2, random_state=5)

#adition
X_train.shape
X_test.shape										

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

print(regressor.intercept_)#For retrieving the slope:
print(regressor.coef_)

#X-test is the file that we want to predict
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

#Data prediction for first 5 data
print(df.head(5))

#Data prediction for last 5 data
print(df.tail(5))


df

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()







X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
y = dataset['quality'].values

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['quality'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

