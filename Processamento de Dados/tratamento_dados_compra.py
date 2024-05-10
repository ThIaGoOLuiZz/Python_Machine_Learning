import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn .impute import SimpleImputer

dataset = pd.read_csv('Processamento de Dados\Data.csv')                                    #Ler o csv.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')                             #Instanciar classe SimpleImputer, passando como parametro o tipo a ser substituido(nan), definindo a strategy(estrategia) como mean(média).

x = dataset.iloc[:, :-1].values                                                             #Definindo o vetor de features(recursos).
y = dataset.iloc[:, -1].values                                                              #Definindo o vetor de Dependent Variable (Variavel dependente).

print("---------------------MATRIZ ORIGINAL-------------------")
print(x)

print("\n---------------------VETOR VARIAVEL DEPENDENTE-------------------")
print(y)

imputer.fit(x[:, 1:3])                                                                      #Define o fit(tamanho) do imputer.
x[:, 1:3] = imputer.transform(x[:, 1:3])                                                    #Atribui os valores atualizados com a média para o mesmo range selecionado no fit.

print("\n---------------------SUBTITUICAO DOS VALORES 'nan'-------------------")
print(x)