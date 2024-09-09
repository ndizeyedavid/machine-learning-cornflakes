# necessarities
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# outpuuting the modal library
from joblib import dump

# training data

# generating training years
years = np.arange(1979, 2024).reshape(-1, 1)
# print(years)

# using seed for reproducibility and preventing auto refreshing of training data
np.random.seed(42)

# generated Dropouts values
dropouts = 15000 - 200 * (years - 2000) + np.random.randint(-1000, 1000, size=years.shape) 
# print(dropouts)

# plot the data on a scater graph
# plt.scatter(years, dropouts)
# plt.title("Yearly dropouts from 2000 - 2024")
# plt.xlabel("Year")
# plt.ylabel("Dropouts")
# plt.show()


# splitting and training the data to the model
x_train, x_test, y_train, y_test = train_test_split(years, dropouts, test_size=0.2, random_state=0)

# create a model and train it with that data
model = LinearRegression()
model.fit(x_train, y_train)

# save the model separately
# dump(model, '../models/school_dropouts.joblib')
# print("Model saved successfully");


# make some prediction form the testing data we splitted
y_pred = model.predict(x_test)

# calulate the uncertainty
mse = mean_squared_error(y_test, y_pred)
print(f"The mean squared error = {mse}")

# Verify the outcome
r_squared = model.score(x_test, y_test)
print(f"R-Squared: {r_squared}")


# make another graph
plt.scatter(x_test, y_test, color="blue", label="Actual value")
plt.plot(x_test, y_pred, color="red", label="Prediction")
plt.xlabel("Years")
plt.ylabel("Predictions")
plt.legend()
plt.show()

