# Import Dataset and Libraries
from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt 
# import kmeans module
from k_means import KMeans

# Others
import pandas as pd
import numpy as np

# # centoids
# centroids = [(-5,-5),(5,5),(-2.5,2.5),(2.5,-2.5)]
# cluster_std = [1,1,1,1] 

# # Create Dataset
# X, y = make_blobs(
#     n_samples=100, 
#     cluster_std=cluster_std,
#     centers=centroids,
#     n_features=2, 
#     random_state=2
# )


# Instead of Generated Data
df = pd.read_csv('Day 70 - K-means clustering\student_clustering.csv')

X = df.iloc[:,:].values
# Check wheater actually cluster forming
# plt.scatter(data.iloc[:,0], data.iloc[:,-1])

km = KMeans(n_clusters=4, max_iter=100)
y_means = km.fit_predict(X)


# plot scatter
plt.scatter(X[y_means == 0,0], X[y_means == 0,1], color='red')
plt.scatter(X[y_means == 1,0], X[y_means == 1,1], color='blue')
plt.scatter(X[y_means == 2,0], X[y_means == 2,1], color='green')
plt.scatter(X[y_means == 3,0], X[y_means == 3,1], color='olive')
# # show plot
plt.show()