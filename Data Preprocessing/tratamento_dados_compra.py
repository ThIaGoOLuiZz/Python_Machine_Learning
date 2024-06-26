import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Data Preprocessing\Data.csv')                                        #Ler o csv.

x = dataset.iloc[:, :-1].values                                                             #Definindo o vetor de features(recursos).
y = dataset.iloc[:, -1].values                                                              #Definindo o vetor de Dependent Variable (Variavel dependente).

print("---------------------MATRIZ ORIGINAL-------------------")
print(x)

print("\n---------------------VETOR VARIAVEL DEPENDENTE-------------------")
print(y)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')                             #Instanciar classe SimpleImputer, passando como parametro o tipo a ser substituido(nan), definindo a strategy(estrategia) como mean(média).
imputer.fit(x[:, 1:3])                                                                      #Define o fit(tamanho) do imputer.
x[:, 1:3] = imputer.transform(x[:, 1:3])                                                    #Atribui os valores atualizados com a média para o mesmo range selecionado no fit.

print("\n---------------------SUBTITUICAO DOS VALORES 'nan'-------------------")
print(x)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') #Instancia classe para transformar coluna em hotEncoder
x = np.array(ct.fit_transform(x)) #Faz a transformação da coluna
print("\n---------------------HOT-ENCODING-------------------")
print(x)

le = LabelEncoder() #Instancia classe LabelEncoder
y = le.fit_transform(y) #Transforma a coluna Dependent Variable para LabelEncoder
print("\n---------------------LabelEncoder(Transforma no/yes para 0/1)-------------------")
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) #Fazendo o split aleatorio das matrizes x e y. Mantendo 80% para treino e 20% para testes
print("\n---------------------X TRAIN-------------------")
print(x_train)
print("\n---------------------X TEST-------------------")
print(x_test)
print("\n---------------------Y TRAIN-------------------")
print(y_train)
print("\n---------------------Y TEST-------------------")
print(y_test)

sc = StandardScaler() #INSTANCIA STANDARDISATION (PADRONIZAÇÃO) -- COLOCAR OS DADOS NA MESMA ESCALA
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:]) #CALCULA A MEDIA E O DESVIO PADRÃO DO TREINAMENTO
x_test[:, 3:] = sc.transform(x_test[:, 3:]) #UTILIZA A MEDIA E O DESVIO PADRÃO PARA TRANSFORMAR O CONJUNTO DE TESTE

print("\n---------------------PADRONIZAÇÃO TREINO-------------------")
print(x_train)
print("\n---------------------PADRONIZAÇÃO TESTE-------------------")
print(x_test)