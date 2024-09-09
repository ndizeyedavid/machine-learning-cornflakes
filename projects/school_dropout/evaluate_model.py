
# this file uses the model saved and evaluates it on the data provided in the test.csv

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib

# load tesing data

testing_data = pd.read_csv('test.csv')

model = joblib.load('school_dropout_model.joblib')

y_pred = model.predict(testing_data)

print(y_pred)