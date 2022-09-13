import numpy as np
import pandas as pd
import math
import os
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('C:/Users/Vitor/Desktop/titanic/train.csv')  # recebe o database
test = pd.read_csv('C:/Users/Vitor/Desktop/titanic/test.csv')  # recebe os casos de teste

Y = train["Survived"]

features = ["Pclass", "Sex", "Parch"]  # gostaria de usar Age tbm, porem nao consegui arrumar o erro que causa
X = pd.get_dummies(train[features])  # onehot
X_test = pd.get_dummies(test[features])  # onehot

model = RandomForestClassifier(random_state=1)

model.fit(X, Y)
predictions = model.predict(X_test)  # realiza as previsoes

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)