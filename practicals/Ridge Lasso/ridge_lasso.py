import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn import datasets

boston_dataset = datasets.load_boston()
boston_pd = pd.DataFrame(boston_dataset.data)
boston_pd.columns = boston_dataset.feature_names
boston_pd_target = np.asarray(boston_dataset.target)
boston_pd['House Price'] = pd.Series(boston_pd_target)
 
# input
X = boston_pd.iloc[:, :-1]
 
#output
Y = boston_pd.iloc[:, -1]

# print(boston_pd.head())

x_train, x_test, y_train, y_test = train_test_split(boston_pd.iloc[:, :-1], boston_pd.iloc[:, -1],test_size = 0.25)
 
# print("Train data shape of X = % s and Y = % s : "%(x_train.shape, y_train.shape))
# print("Test data shape of X = % s and Y = % s : "%(x_test.shape, y_test.shape))

lreg = LinearRegression()
lreg.fit(x_train, y_train)
 
# Generate Prediction on test set
lreg_y_pred = lreg.predict(x_test)
 
# calculating Mean Squared Error (mse)
mean_squared_error = np.mean((lreg_y_pred - y_test)**2)
# print("Mean squared Error on test set : ", mean_squared_error)

# Putting together the coefficient and their corresponding variable names
lreg_coefficient = pd.DataFrame()
lreg_coefficient["Columns"] = x_train.columns
lreg_coefficient['Coefficient Estimate'] = pd.Series(lreg.coef_)
# print(lreg_coefficient)


from sklearn.linear_model import Ridge
 
# Train the model
ridgeR = Ridge(alpha = 1)
ridgeR.fit(x_train, y_train)
y_pred = ridgeR.predict(x_test)
 
# calculate mean square error
mean_squared_error_ridge = np.mean((y_pred - y_test)**2)
# print(mean_squared_error_ridge)

# get ridge coefficient and print them
ridge_coefficient = pd.DataFrame()
ridge_coefficient["Columns"]= x_train.columns
ridge_coefficient['Coefficient Estimate'] = pd.Series(ridgeR.coef_)
# print(ridge_coefficient)

from sklearn.linear_model import Lasso
 
# Train the model
lasso = Lasso(alpha = 1)
lasso.fit(x_train, y_train)
y_pred1 = lasso.predict(x_test)
 
# Calculate Mean Squared Error
mean_squared_error = np.mean((y_pred1 - y_test)**2)
# print("Mean squared error on test set", mean_squared_error)

lasso_coeff = pd.DataFrame()
lasso_coeff["Columns"] = x_train.columns
lasso_coeff['Coefficient Estimate'] = pd.Series(lasso.coef_)
 
# print(lasso_coeff)

# from sklearn.decomposition import PCA

# pca = PCA()
# X_reduced = pca.fit_transform(X)

# print(X_reduced)
# x_df = pd.DataFrame(pca.components_.T)

# lreg.fit(X_reduced, y_train)

# # Generate Prediction on test set
# lreg_y_pred = lreg.predict(x_test)
 
# # calculating Mean Squared Error (mse)
# mean_squared_error = np.mean((lreg_y_pred - y_test)**2)
# # print("Mean squared Error on test set : ", mean_squared_error)

# # Putting together the coefficient and their corresponding variable names
# lreg_coefficient = pd.DataFrame()
# lreg_coefficient["Columns"] = x_df.columns
# lreg_coefficient['Coefficient Estimate'] = pd.Series(lreg.coef_)