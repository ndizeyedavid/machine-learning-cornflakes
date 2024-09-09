
# this file provides accuracy, performance report sheet for this model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('dataset.csv')

X = data.drop('Output', axis=1)
y= data['Output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)

model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy = {accuracy}")
print("================================= Report ==================================")
print (report)