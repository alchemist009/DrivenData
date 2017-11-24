import numpy as np
import pandas as pd

train_label = pd.read_csv("Training_set_labels.csv")

column_labels = list(train_label.columns.values)
column_labels.remove("id")

for i in column_labels:
	unique_value = train_label[i].unique()
	size = len(unique_value)
	print(size)
	for j in range(size):
		if unique_value[j] != "nan":
			train_label.loc[train_label[i] == unique_value[j], i] = j

train_label.to_csv("train_label_clean.csv", index = False)

# training_labels = np.genfromtxt("Training_set_labels.csv", dtype=None,  delimiter=',')
# print training_labels

# training_values = np.genfromtxt("Training_set_values2.csv", dtype=None, delimiter=',')
# print training_values
