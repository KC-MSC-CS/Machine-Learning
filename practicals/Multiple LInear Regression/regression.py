import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

bh_data = load_boston()

print(bh_data.keys())

boston = pd.DataFrame(bh_data.data, columns =bh_data.feature_names)
boston['MEDV'] = bh_data.target

boston_df = pd.DataFrame(bh_data.data)
boston_df.columns = bh_data.feature_names
boston_df.head()

boston_df.isnull().sum()

X = boston_df
Y = bh_data.target

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state = 9)
model = LinearRegression()
model.fit(x_train,y_train)

pred = model.predict(x_test)

print(model.score(X,Y))
print('Coefficient: ', model.coef_)
print('Intercept: ',model.intercept_)

test_set_rmse = (np.sqrt(mean_squared_error(y_test,pred)))
test_set_r2 = r2_score(y_test, pred)

print("Test set RMSE: ", test_set_rmse)
print("Test set R2: ", test_set_r2)