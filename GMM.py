import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import pandas as pd



Data1=pd.read_csv("aligned_trajectory_data_z.csv")
index = Data1.index.values
x = Data1[["Aligned_X"]].values
y = Data1[["Aligned_Y"]].values

# Reshape to 2D array (n_samples, n_features)
combine_data=np.column_stack((x, y))



# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=4)
gmm.fit(combine_data)

# Predict cluster labels
labels = gmm.predict(combine_data)

plt.scatter(x,y, c=labels, cmap='viridis', alpha=0.6)
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Aligned_X")
plt.ylabel("Aligned_Y")
#plt.plot(y)
plt.show()

plt.scatter(index,y, c=labels, cmap='viridis', alpha=0.6)
plt.title("GMM Clustering on Aligned_X vs Index")
plt.xlabel("Index")
plt.ylabel("Aligned_Y(Reference)")
plt.plot(y)
plt.show()

plt.scatter(index,x, c=labels, cmap='viridis', alpha=0.6)
plt.title("GMM Clustering on Aligned_Z vs Index")
plt.xlabel("Index")
plt.ylabel("Aligned_X(Query)")
plt.plot(x)
plt.show()