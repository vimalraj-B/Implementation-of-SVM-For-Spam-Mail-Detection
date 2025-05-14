# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import libraries.
2.Read the CSV file and display data using head().
3.Split the dataset using train_test_split().
4.Calculate predictions and accuracy.
5.Print the outputs.
6.End the program.
```
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VIMALRAJ B
RegisterNumber:  212224230304
*/

import chardet
file=(r'C:\Users\admin\Downloads\spam.csv')
with open(file,'rb')as rawdata:
    result=chardet.detect(rawdata.read(100000))
print(result)
import pandas as pd
data=pd.read_csv(r'C:\Users\admin\Downloads\spam.csv',encoding='Windows-1252')
print(data.head())
print(data.info())
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)

```

## Output:

```
ENCODING DETECTED
```
![Screenshot 2025-05-14 111550](https://github.com/user-attachments/assets/3c174a0d-e1c0-4bfe-bb93-c1a9051b084a)

```
FIRST FEW ROWS,DATA INFO,MISSING VALUES
```
![Screenshot 2025-05-14 111746](https://github.com/user-attachments/assets/87820926-6a75-4113-8ec4-d802e58345ae)

```
PREDICTED LABELS
```
![Screenshot 2025-05-14 111831](https://github.com/user-attachments/assets/b074448a-b232-4602-8ff2-913ebdb66ea2)


```
MODEL ACCURACY
```

![Screenshot 2025-05-14 111836](https://github.com/user-attachments/assets/b95f2692-4a38-4c31-8e89-17f236aca0b4)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
