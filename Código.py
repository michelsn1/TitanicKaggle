# -*- coding: utf-8 -*-
"""
Created on Tue May  8 21:45:42 2018

@author: miche
"""

import pandas as pd
import numpy as np

training_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")

training_set.drop(['Name','Ticket','Cabin' ], axis=1, inplace =True)
test_set.drop(['Name','Ticket','Cabin' ], axis=1, inplace =True)

train=pd.get_dummies(training_set)
teste=pd.get_dummies(test_set)

train.isnull().sum().sort_values(ascending=False).head(10)


#Criando os vetores de entrada e saída
X = train.iloc[:, 2:13].values
y = train.iloc[:, 1].values


# Trabalhando com a possibilidade de falta de dados em X
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:10])
X[:, 1:10] = imputer.transform(X[:, 1:10])

teste.isnull().sum().sort_values(ascending=False).head(10)
teste['Age'].fillna(value = teste['Age'].mean(),inplace =True)
teste['Fare'].fillna(value = teste['Fare'].mean(),inplace =True)

#Dividindo os dados em Training set e Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Ajustando o modelo de Random Forest Regression
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
classifier.score(X_test,y_test)
#Observando-se os resultados previstos
y_pred = classifier.predict(X_test)

submission=pd.DataFrame()
submission['PassengerId']= teste['PassengerId']
submission['Survived']=classifier.predict(teste.values[:,1:11])
submission.to_csv('submission.csv',index=False)
