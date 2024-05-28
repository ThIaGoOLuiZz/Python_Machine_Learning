import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Regressions\Polynomial Regression\Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y),1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)

regressor = SVR(kernel='rbf')
regressor.fit(x_train, y_train)

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(x_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
