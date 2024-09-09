
# nothing good like training this model with well formatted and finilised training data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data = pd.read_csv('dataset.csv')

X = data.drop('Output', axis=1)
y= data['Output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=48)

model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)

joblib.dump(model, "school_dropout_model.joblib")

print("Model trainied successfully")