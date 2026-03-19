# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect and preprocess email data by cleaning text, removing stop words, and converting messages into numerical features (like TF-IDF).
2. Label the dataset as spam or not spam and split it into training and testing sets.
3. Train a Support Vector Machine (SVM) model using the training data to learn classification boundaries.
4. Use the trained SVM model to predict whether new emails are spam or not.
5. Evaluate the model performance using accuracy, precision, recall, or F1-score.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Jayasri L
RegisterNumber: 212224040136  
*/
```
```
import chardet
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv", encoding='Windows-1252')

data.head()
data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```
## Output:

<img width="866" height="914" alt="image" src="https://github.com/user-attachments/assets/27228b69-0e43-4c29-9966-e217855fd9de" />

<img width="867" height="303" alt="image" src="https://github.com/user-attachments/assets/3459c1e0-8d6b-4323-8f45-da7e647bc970" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
