# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 05:26:10 2018

@author: praveen
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

train=pd.read_csv('C:\\Users\\praveen\\Desktop\\titanic data\\train.csv')
test=pd.read_csv('C:\\Users\\praveen\\Desktop\\titanic data\\test.csv')

#train.shape
#x=train[['Pclass','Age','Sex', 'SibSp', 'Parch', 'Embarked']]
#x_1=x.iloc[:,:-1]
'''x=train.iloc[:,5:8].values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NAN',strategy='mean',axis=0)
imputer.fit(x[:,0:2])
x[:,0:2]=imputer.transform(x[:,0:2])
'''
#len(train['Sex'])
#print(len(train['Age']))
'''
train['Sex'][2]
for i in range(len(train['Age'])):
    if train['Age'][i]=='male':
        train['Age'][i]=0
    else:
        train['Age'][i]=1
   '''

def clean_data(data):
    data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
    data['Age'] =  data['Age'].fillna(data['Age'].dropna().median())
    
    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] =='female',  'Sex'] = 1
    
    data['Embarked'] = data['Embarked'].fillna('S')
    data.loc[data["Embarked"] == 'S', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] =2

    
    
    
def write_prediction(prediction, name):
    PassengerId = np.array(test['PassengerId']).astype(int)
    solution = pd.DataFrame(prediction, PassengerId, columns = ['Survived'])
    solution.to_csv(name, index_label = ['PassengerId'])
    
    

clean_data(train)
clean_data(test)
x_train=train[['Sex','Age','Embarked']].values

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x_train[:,0:3])
x_train[:,0:3]=imputer.transform(x_train[:,0:3])

target=train['Survived'].values
#features_forest = train[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Embarked']].values

#feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
target=sc_x.fit(target)
'''
#logistic regression
'''from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,target)

y_pred=classifier.predict(target)
'''
#Random forest

from sklearn import ensemble, model_selection
forest = ensemble.RandomForestClassifier(
        max_depth = 7,
        min_samples_split = 4,
        n_estimators = 100,
        n_jobs = -1,
       random_state = 1
        )
forest = forest.fit(x_train, target)



print(forest.feature_importances_)
print(forest.score(x_train, target))

scores = model_selection.cross_val_score(forest, x_train, target, scoring = 'accuracy', cv = 10)

print(scores.mean())


test_features_forest = test[['Sex','Age','Embarked']].values
prediction_forest = forest.predict(test_features_forest)
write_prediction(prediction_forest, "C:\\Users\\praveen\\Desktop\\machine-learning udemy a-z\\Machine Learning A-Z Template Folder\\Part 8 - Deep Learning\\Section 39 - Artificial Neural Networks (ANN)\\Artificial_Neural_Networks\\resultsrandom_forest2.csv")



























