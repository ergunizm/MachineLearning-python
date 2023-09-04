import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

lin_reg = LinearRegression()
lin_reg.fit(x,y)

poly_reg = PolynomialFeatures(degree=4) #max degree
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

plt.subplot(2,1,1)
plt.scatter(x,y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salary')

plt.subplot(2,1,2)
plt.scatter(x,y, color='red')
plt.plot(x, lin_reg2.predict(x_poly), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Level')
plt.ylabel('Salary')

plt.show()

'''(for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Polynomial Regression(Smooth)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
'''

s_lin_pred = lin_reg.predict([[6.5]])
s_poly_pred = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))

print(s_lin_pred, s_poly_pred)