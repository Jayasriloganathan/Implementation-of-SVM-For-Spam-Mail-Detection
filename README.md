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

with open('spam.csv', 'rb') as file:
    encoding = chardet.detect(file.read(100000))
print(encoding)


import pandas as pd

data = pd.read_csv('spam.csv', encoding='Windows-1252')

print(data.head())
print(data.isnull().sum())


X = data['v2']   
y = data['v1']  


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Predicted values:", y_pred)


from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))

```
## Output:

<img width="1542" height="627" alt="image" src="https://github.com/user-attachments/assets/cd32fec5-6406-40ac-ad81-f7ba18f2a45d" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
