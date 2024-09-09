# i don't know know how this works, it's like this "iris" thing is a novel / flower idk, i just want to knwo more about Ai that's all🥹

# found it!! :) they are pre-built datasets in this sklearn library
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# print(f"Accuracy  = {accuracy}")
# print("classification report:")
print(X)