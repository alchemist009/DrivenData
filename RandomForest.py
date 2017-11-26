
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[2]:

train = pd.read_csv("train_set.csv")
test = pd.read_csv("test_set.csv")


# In[3]:

dummy_cols = ['funder', 'installer', 'basin', 'public_meeting', 'scheme_management'
             , 'permit', 'construction_year', 'extraction_type_class', 'payment', 
              'water_quality', 'quantity', 'source', 'source_class', 'waterpoint_type',
             'waterpoint_type_group']

train = pd.get_dummies(train, columns= dummy_cols)

train = train.sample(frac=1).reset_index(drop=True)


# In[4]:

test = pd.get_dummies(test, columns=dummy_cols)


# In[5]:

target = train.status_group
features = train.drop('status_group', axis=1)

X_train, X_val, y_train, y_val = train_test_split(features, target, train_size=0.8)


# In[ ]:

def model_random_forest(X_train, X_val, y_train, y_val, test):
    if __name__ == '__main__':
    
        rf = RandomForestClassifier(criterion='gini',
                                   max_features='auto',
                                   min_samples_split=6,
                                   oob_score=True,
                                   random_state=1,
                                   n_jobs=-1)

        param_grid = {'n_estimators' : [500,750,1000]}

        gs = GridSearchCV(estimator=rf,
                         param_grid=param_grid,
                         scoring='accuracy',
                         cv=2,
                         n_jobs=-1)

        gs = gs.fit(X_train, y_train)

        best_params = gs.best_params_
        cv_results = gs.cv_results_
        validation_accuracy = gs.score(X_val, y_val)

        print("Validation accuracy: ", validation_accuracy)
        print(best_params)
        print(cv_results)
    


# In[ ]:

model_random_forest(X_train, X_val, y_train, y_val, test)

