import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv(r'Regressions/Decision Tree Regression/Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

regressor.predict([[6.5]])

print(regressor.predict([[6.5]]))

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Verdade ou mentira (Decision Tree Regression)')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()