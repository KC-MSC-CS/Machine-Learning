import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


x = np.array([1, 3, 10, 16, 26, 36]).reshape(-1, 1)
y = np.array([42, 50, 75, 100, 150, 200]).reshape(-1, 1)

lr = LinearRegression()
lr.fit(x, y)

m = lr.coef_
c = lr.intercept_

print("Slope: ", m, " Intercept: ", c)

y_pred = lr.predict([[2], [3], [5], [11], [13], [37]])

plt.scatter(x, y)
plt.plot([2, 3, 5, 11, 13, 37], y_pred, color='k')
plt.show()
