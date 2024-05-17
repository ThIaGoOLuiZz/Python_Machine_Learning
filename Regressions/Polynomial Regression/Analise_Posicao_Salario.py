import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Regressions\Polynomial Regression\Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

lin_reg = LinearRegression()
lin_reg.fit(x,y)

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# LINEAR REGRESSION
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Verdade ou mentira (Linear Regression)')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

# POLYNOMIAL REGRESSION
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Verdade ou mentira (Polynomial Regression)')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

print(lin_reg.predict([[6.5]])) #POSIÇÃO 6.5 // LINEAR REGRESSION

print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))) #POSIÇÃO 6.5 // POLYNOMIAL REGRESSION