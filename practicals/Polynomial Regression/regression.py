import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.metrics import mean_squared_error

# Importing the dataset
dataset = pd.read_csv("./Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

print(dataset.head())

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

y_pred_1 = lin_reg.predict(X)

rmse = (np.sqrt(mean_squared_error(y, y_pred_1)))
print("RMSE ",rmse)

# Visualizing the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

y_poly2 = pol_reg.predict(poly_reg.fit_transform(X))

rmse_poly = (np.sqrt(mean_squared_error(y, y_poly2)))
print("RMSE ",rmse_poly)

# Visualizing the Polymonial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

# Visualizing the Polymonial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


y_polyP = pol_reg.predict(poly_reg.fit_transform(X))
rmse_poly = (np.sqrt(mean_squared_error(y, y_polyP)))
print("RMSE ",rmse_poly)
