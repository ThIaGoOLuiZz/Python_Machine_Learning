import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Processamento de Dados\Data.csv')

x = dataset.iloc[:, :-1].values     #Definindo as Caracteristicas
y = dataset.iloc[:, -1].values      #Definindo a Dependent Variable (Variavel dependente)

print(x)
print(y)