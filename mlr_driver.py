"""
Multiple linear regression model
using processed data set with y value being storm counts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from Modeling_Tools import model_load
from Modeling_Tools import model_save
from Modeling_Tools import calculate_metrics

# Load model if it exists
model = model_load('MLR.pkl')

# Load data set
data = pd.read_csv('processed_dat.csv')
df_cleaned = data.dropna()
X_labels = ['Field Magnitude Average |B|', 'f10.7_index', 'Proton Density', 'Flow Pressure',
            'Plasma (Flow) speed', 'Proton temperature', 'Na/Np', 'R', 'DST Index Min']

# Assign the data and target
X = pd.DataFrame(df_cleaned, columns=X_labels)
y = pd.Series(df_cleaned['total storms'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
if model is None:
    model = LinearRegression()
    model.fit(X_train, y_train)
    model_save(model, name='MLR.pkl')

# Make Predictions
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Make Scatter Plot
ax = sns.scatterplot(x=y_test, y=y_pred)
ax = sns.scatterplot(x=y_train, y=y_pred_train)

#generate line
m = 1  # Slope
c = 0  # Intercept
x_line = np.array([0, 500])
y_line = m * x_line + c

#generate metrics
#test
r_squared_test, mse_test = calculate_metrics(y_test, y_pred)
print(f"R-squared for test: {r_squared_test}")
print(f"MSE for test: {mse_test}")
#train
r_squared_train, mse_train = calculate_metrics(y_train, y_pred_train)
print(f"R-squared for train: {r_squared_train}")
print(f"MSE for train: {mse_train}")

plt.plot(x_line, y_line, color='black') # You can change the color as needed

ax.set(xlabel="Actual Values", ylabel="Predicted Values")
plt.title("MLR Actual vs Predicted Values")
ax.legend(['Testing', 'Training'])
plt.show()



