# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries in python for finding linear regression.
2. Assign a variable 'dataset' for reading the content in given csv file.
3. Split the supervised data and unsupervised data in the dataset using train_test_split method.
4. Using training and test values on dataset, predict the linear line.
5. Assign the points for representing in the graph.
6. Predict the regression for marks by using the rep resentation of the graph.
7. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

'''
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Senthil Kumar S
RegisterNumber: 212221230091
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
df=pd.read_csv('student_scores - student_scores.csv')
df.head()
df.tail()

x=df.iloc[:,:-1].values
y=df.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

# For training data

plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours vs Scores(Train)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# For testing data

plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours vs Scores(Test)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
### Contents in the data file (head, tail):
![o1](https://user-images.githubusercontent.com/93860256/162620331-6ba93613-d82e-412c-a4ba-c60f7aef094f.png)
![o2](https://user-images.githubusercontent.com/93860256/162620342-208ba19a-41bc-4ebc-9f37-581b6e72248b.png)
### Linear Regression Graph(For Training Data):
![o3](https://user-images.githubusercontent.com/93860256/162620285-001e6218-5999-4a6f-8b65-b1bdea94fa06.PNG)
### Linear Regression Graph(For Test Data):
![o4](https://user-images.githubusercontent.com/93860256/162620302-5fb18f0d-ae93-4735-a976-f049ad5ffa11.PNG)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
