import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Regressions\Simple Linear Regression\Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

regressor = LinearRegression() #INSTANCIA CLASSE DE REGRASSÃO LINEAR (LINEAR REGRESSION)
regressor.fit(x_train, y_train) #MODELO APRENDE COMO OS DADOS ESTÃO RELACIONADOS

y_pred = regressor.predict(x_test) #FAZ UMA PREVISÃO

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salario X Experiencia (Trainning set)')
plt.xlabel('Anos de Experiencia')
plt.ylabel('Salario')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salario X Experiencia (Test set)')
plt.xlabel('Anos de Experiencia')
plt.ylabel('Salario')
plt.show()