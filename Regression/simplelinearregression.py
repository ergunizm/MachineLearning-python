import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

plt.subplot(1,2, 1)
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Training Set')
plt.xlabel('Experience')
plt.ylabel('Salary')

plt.subplot(1,2, 2)
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')  #ikisi de aynı doğruyu vereceği için değiştirmedik
plt.title('Test Set')
plt.xlabel('Experience')
plt.ylabel('Salary')

plt.show()
print(regressor.predict([[12]]))
print(regressor.coef_)
print(regressor.intercept_)