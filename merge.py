import glob, os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

a = pd.read_csv("C:/Users/wra1th/Desktop/Projects/DrivenData/Training_set_labels.csv")
b = pd.read_csv("C:/Users/wra1th/Desktop/Projects/DrivenData/Training_set_values.csv")

results = a.merge(b, on='id')

results.to_csv('C:/Users/wra1th/Desktop/Projects/DrivenData/Training_set.csv', index=False)