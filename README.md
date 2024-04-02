# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. .
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SHANMUGA PRIYA.T
RegisterNumber: 212222040153 
*/

import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
## DATASET

![image](https://github.com/shanmugapriyatharani/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393427/cf71c54f-70dc-486d-9682-b048c223ed8e)

## data.info()

![image](https://github.com/shanmugapriyatharani/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393427/8491c459-bc49-46f8-9753-c7cb9898bab0)

## CHECKING IF NULL VALUES ARE PRESENT

![image](https://github.com/shanmugapriyatharani/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393427/3b5e4889-23e0-4204-8f8a-f643a47c21d9)

## VALUE_COUNTS()

![image](https://github.com/shanmugapriyatharani/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393427/a64d1d14-8480-4ec5-946a-d33efe21b063)

## DATASET AFTER ENCODING

![image](https://github.com/shanmugapriyatharani/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393427/e41fa246-7c40-4a30-b82f-eacc561f2060)

## X-VALUES

![image](https://github.com/shanmugapriyatharani/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393427/34713e1d-032b-4fc4-9746-25764235ee69)

## ACCURACY

![image](https://github.com/shanmugapriyatharani/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393427/33de318f-3040-441e-b449-3ad9f1313518)

## dt.predict()

![image](https://github.com/shanmugapriyatharani/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393427/a064f8b3-3766-406e-b3d8-66444c24ded3)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
