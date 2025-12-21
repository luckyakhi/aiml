import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# 1. Load Data
iris = datasets.load_iris()
X = iris.data

# Note: We are NOT loading 'y' (target). The model won't know 'Setosa' exists.

# 2. Instantiate K-Means
# We tell it to find 3 clusters (because we secretly know there are 3 species)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# 3. Fit
# Notice we only pass 'X'. No 'y'!
kmeans.fit(X)

# 4. Get the results
# The labels_ are the group numbers (0, 1, 2) the model assigned
predicted_labels = kmeans.labels_

print("--- The Model Found These Groups ---")
print(predicted_labels)

# Create a plot comparing Actual Species vs. K-Means Clusters
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Actual Species (The Truth)
axes[0].scatter(X[:, 2], X[:, 3], c=iris.target, cmap='viridis', s=50)
axes[0].set_title("Actual Species (Ground Truth)")
axes[0].set_xlabel("Petal Length")
axes[0].set_ylabel("Petal Width")

# Plot 2: K-Means Clusters (The Discovery)
axes[1].scatter(X[:, 2], X[:, 3], c=predicted_labels, cmap='viridis', s=50)
axes[1].set_title("K-Means Clusters (Unsupervised)")
axes[1].set_xlabel("Petal Length")
axes[1].set_ylabel("Petal Width")

# Plot the Centroids (The big red X's)
centers = kmeans.cluster_centers_
axes[1].scatter(centers[:, 2], centers[:, 3], c='red', s=200, marker='X', label='Centroids')
axes[1].legend()

plt.show()