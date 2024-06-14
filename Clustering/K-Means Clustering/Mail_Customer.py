import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Clustering\Dataset\Mall_Customers.csv')

x = dataset.iloc[:, [3,4]].values