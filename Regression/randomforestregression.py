import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x,y)
print(regressor.predict([[6.5]]))

x_grid = np.arange(min(x), max(x), 0.1)
x_grid =x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()