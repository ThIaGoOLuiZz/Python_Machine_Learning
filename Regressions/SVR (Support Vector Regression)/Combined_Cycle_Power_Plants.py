import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

dataset = pd.read_csv('Regressions\SVR (Support Vector Regression)\Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y),1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)

regressor = SVR(kernel='rbf')
regressor.fit(x_train, y_train.ravel())

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(x_test)).reshape(-1, 1))

np.set_printoptions(precision=2)
print(np.concatenate((y_pred, y_test), axis=1))

print(r2_score(y_test, y_pred))