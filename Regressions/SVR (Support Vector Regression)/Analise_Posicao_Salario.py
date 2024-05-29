import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv(r'Regressions/SVR (Support Vector Regression)/Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = y.reshape(-1, 1)
y = sc_y.fit_transform(y).ravel()

regressor = SVR(kernel='rbf')
regressor.fit(x, y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1, 1))
print(y_pred)

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y.reshape(-1, 1)), color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)), color='blue')
plt.title('Verdade ou mentira (SVR)')
plt.xlabel('Nível')
plt.ylabel('Salário')
plt.show()