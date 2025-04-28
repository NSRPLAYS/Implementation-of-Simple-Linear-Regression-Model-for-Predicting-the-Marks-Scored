# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NAVEEN SAIRAM B
RegisterNumber: 24009928

regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
mse=mean_squared_error(y_test,y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,y_pred,color="green")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
*/
```

## Output:
## head value
![image](https://github.com/user-attachments/assets/b713db7f-00aa-42e1-b43e-8ff2f5abe776)
## X values
![image](https://github.com/user-attachments/assets/e10a4da9-52d9-45f9-9c70-5adbbd277490)
## Y values
![image](https://github.com/user-attachments/assets/c4f6d179-3701-437e-b18c-99bc40a5d0a2)
## TRAINING SET
![image](https://github.com/user-attachments/assets/8f663c66-2630-4338-b34f-b3ef93ccdcd7)
## TESTING SET
![image](https://github.com/user-attachments/assets/225b4f68-e6d7-49ca-83db-909eea8a7485)
## ERROR:
![image](https://github.com/user-attachments/assets/9c3a1527-71a8-4285-8251-6b5ae0d5688e)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
