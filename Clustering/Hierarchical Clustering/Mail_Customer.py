import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch

dataset = pd.read_csv('Clustering\Dataset\Mall_Customers.csv')

x = dataset.iloc[:, [3,4]].values

dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()


# plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1], s = 100, c='red', label = 'Cluster 1')
# plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], s = 100, c='blue', label = 'Cluster 2')
# plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], s = 100, c='green', label = 'Cluster 3')
# plt.scatter(x[y_kmeans == 3,0], x[y_kmeans == 3,1], s = 100, c='cyan', label = 'Cluster 4')
# plt.scatter(x[y_kmeans == 4,0], x[y_kmeans == 4,1], s = 100, c='Magenta', label = 'Cluster 5')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label= 'Centroids')
# plt.title('Clusters of Customers')
# plt.xlabel('Annual Income (K$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show()