import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split as tts

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values   #features
y = dataset.iloc[:, -1].values #dependent variable
# print(x)
# print(y)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])  #transform returns new not replaces
# print(x)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])] ,remainder='passthrough') #remainder yapmazsak işlem görmeyen satırlar kaybolur
x = np.array(ct.fit_transform(x))
# print(x)

le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=1)
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)