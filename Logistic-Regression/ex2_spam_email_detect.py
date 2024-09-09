import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("../datasets/spam_email.csv")
testing_data = pd.read_csv("spam_email_testing.csv")

# Preprocessing
X = data.drop("is_spam", axis=1)
y = data['is_spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)

# testing data
X_test = testing_data.drop("is_spam", axis=1)
y_test = testing_data['is_spam']

y_pred = model.predict(X_test) 


print(y_pred)

accuracy_score = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy score = {accuracy_score}")

print("============================================================Report============================================================")
print(report)


