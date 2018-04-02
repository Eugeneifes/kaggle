__author__ = 'Eugene'
import csv

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier



gender_submission = pd.read_csv('gender_submission.csv', usecols=["Survived"])
train = pd.read_csv('train.csv', usecols=["Sex", "Survived"])
test = pd.read_csv('test.csv', usecols=["Sex"])

#PassengerId
#Survived
#Pclass
#Name
#Sex
#Age
#SibSp
#Parch
#Ticket
#Fare
#Cabin
#Embarked

train = train.dropna()
train_array = train.values


X_train = train_array[:, 1:]
Y_train = train_array[:,  0]

X_train = X_train.tolist()
Y_train = Y_train.tolist()


for i, elem in enumerate(X_train):
    if elem[0] == 'male':
        X_train[i][0] = 0
    else:
        X_train[i][0] = 1

neigh = KNeighborsClassifier()
neigh.fit(X_train, Y_train)

test = test.dropna()
test_array = test.values
X_test = test_array.tolist()

for i, elem in enumerate(X_test):
    if elem[0] == 'male':
        X_test[i][0] = 0
    else:
        X_test[i][0] = 1

prediction = neigh.predict(X_test)


submission = gender_submission.values
submission = submission[:, 0].tolist()


yes = 0
no = 0
for i in range(len(prediction)):
    if submission[i] == prediction[i]:
        yes += 1
    else:
        no += 1
print(yes, no)



