import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

boston = load_boston()
print(boston.keys())

x = boston.data
y = boston.target

df = pd.DataFrame(x, columns=boston.feature_names)
df['TARGET'] = y

x = np.array(df['RM']).reshape(-1, 1)
y = np.array(df['TARGET']).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

model = LinearRegression()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print("Model score: ", score)

y_pred = model.predict(x_test)


plt.scatter(x_test, y_test, color='b')
plt.plot(x_test, y_pred, color='k')
plt.xlabel("No of rooms")
plt.ylabel('House price')
plt.show()

print(np.sqrt(mean_squared_error(y_test, y_pred)));
