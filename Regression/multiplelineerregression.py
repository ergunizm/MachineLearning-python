import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])] ,remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  #auto avoids dummy variables ie. category, state
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)) #concatenates two vector in one column

print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]])) #single predict

#equations
print(regressor.coef_)
print(regressor.intercept_)

#r2 score
score = r2_score(y_test, y_pred)
print(score)