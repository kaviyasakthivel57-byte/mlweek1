import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

#Import Data:

df = pd.read_csv('ecommerce_sales_data.csv')
print ("First few rows:")
print(df.head(5))

#Data Preprocessing:

X = df[['Sales']]
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Simple Linear Regression using sklearn:

model = LinearRegression()
model.fit(X_train, y_train)
y_pred_sklearn = model.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
print (f"\nMSE (sklearn): {mse_sklearn:.2f}")

# Simple Linear Regression without sklearn(Manual Implementation):

x = df['Sales']. values
y = df['Profit']. values
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_xx = np.sum(x ** 2)

m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
c = (sum_y - m * sum_x) / n
print (f" Profit = {m:.2f} * Sales + {c:.2f}")

y_pred = m * x + c
mse = np.mean((y - y_pred) ** 2)
print (f"Mean Squared Error (MSE): {mse:.2f}")

x_new = float (input ("Enter The Sales "))
y_new = m * x_new + c
print (f"Profit: {y_new:.2f}")

#Compare Result:
print (f"\nMSE comparison - sklearn: {mse_sklearn:.2f}, manual: {mse:.2f}")

