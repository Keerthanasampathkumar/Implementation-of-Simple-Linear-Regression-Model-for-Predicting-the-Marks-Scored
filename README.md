# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KEERTHANA S
RegisterNumber:  212222230066
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,reg.predict(X_train),color="blue")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,reg.predict(X_test),color="black")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
#Dataset

![image](https://github.com/Keerthanasampathkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477890/f8717752-e15c-420d-be19-11dc7557a118)

# Head Values

![image](https://github.com/Keerthanasampathkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477890/eefa32eb-e95a-479d-a91b-23d44b2a8dac)

# Tail Values

![image](https://github.com/Keerthanasampathkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477890/27409f65-70d1-4c3b-a59e-086932e7a982)

# X and Y values

![image](https://github.com/Keerthanasampathkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477890/84b8e3a0-a7d8-4fb8-9e37-fb02ef7c45ac)

# Predication values of X and Y

![image](https://github.com/Keerthanasampathkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477890/4fc8ce16-2052-421a-b1e7-2a7e0fc34422)

# MSE,MAE and RMSE

![ii](https://github.com/Keerthanasampathkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477890/d3c5dfd5-1487-4710-ad83-bdb8783e4efd)

# Training Set

![ff 01](https://github.com/Keerthanasampathkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477890/2299219e-ee61-4956-a57e-ec74d4b47e5b)

# Testing Set

![ff 02](https://github.com/Keerthanasampathkumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477890/cbdb1c46-c517-4a7c-9b86-53d055504e41)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
