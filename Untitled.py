
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv("train_set.csv")
test = pd.read_csv("test_set.csv")


# In[2]:

dummy_cols = ['funder', 'installer', 'basin', 'public_meeting', 'scheme_management'
             , 'permit', 'construction_year', 'extraction_type_class', 'payment', 
              'water_quality', 'quantity', 'source', 'source_class', 'waterpoint_type',
             'waterpoint_type_group']

train = pd.get_dummies(train, columns= dummy_cols)

train = train.sample(frac=1).reset_index(drop=True)


# In[3]:

test = pd.get_dummies(test, columns= dummy_cols)


# In[4]:

target = train.status_group
features = train.drop('status_group', axis=1)

X_train, X_val, y_train, y_val = train_test_split(features, target, train_size=0.8)


# In[5]:

def model(X_train, X_val, y_train, y_val, test):
    if __name__ == '__main__':
        param_grid = {'learning_rate': [0.075, 0.7],
                     'max_depth': [13,14],
                     'min_samples_leaf': [15,16],
                     'max_features': [1.0],
                     'n_estimators': [100,200]}
        
        estimator = GridSearchCV(estimator=GradientBoostingClassifier(),
                                param_grid=param_grid,
                                n_jobs=-1)
        estimator.fit(X_train, y_train)
        
        best_params = estimator.best_params_
        
        print(best_params)
        
        validation_accuracy = estimator.score(X_val, y_val)
        print('Validation accuracy: ', validation_accuracy)


# In[6]:

#model(X_train, X_val, y_train, y_val, test)


#{'max_features': 1.0, 'n_estimators': 100, 'learning_rate': 0.075, 
#'max_depth': 14, 'min_samples_leaf': 15}
#('Validation accuracy: ', 0.79887766554433226)


# In[7]:

test_id = pd.read_csv("SubmissionFormat.csv")
test_id.columns = ['id', 'status_group']
test_id = test_id.id


# In[ ]:

def model_for_submission(features, target, test):
    if __name__== '__main__':
        
        best_params = {'learning_rate': [0.075],
                      'max_depth': [14],
                      'min_samples_leaf': [15],
                      'max_features': [1.0],
                      'n_estimators': [100]}
        
        estimator = GridSearchCV(estimator=GradientBoostingClassifier(),
                                param_grid=best_params,
                                n_jobs=-1)
        estimator.fit(features, target)
        
        predictions = estimator.predict(test)
        data = {'ID': test_id, 'status_group': predictions}
        
        submit = pd.DataFrame(data=data)
        
        vals_to_replace = {2: 'functional', 1: 'functional needs repair',
                          0: 'non functional'}
        
        submit.status_group = submit.status_group.replace(vals_to_replace)
        
        submit.to_csv('predictions_gradient_boostin')


# In[ ]:

model_for_submission(features, target, test)

