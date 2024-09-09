import numpy as np
from joblib import load

loaded_model = load('school_dropouts.joblib')

predictions = loaded_model.predict([[2024], [1026]])

print(predictions)