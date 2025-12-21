import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# 1. Load Data
iris = datasets.load_iris()
X = iris.data
#Finding the elbow point
inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Error)')
plt.show()