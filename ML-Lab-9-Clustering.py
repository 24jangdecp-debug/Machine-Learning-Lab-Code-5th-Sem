
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

X, _ = make_blobs(
    n_samples=240, centers=4, random_state=42
)

# K-Means
km = KMeans(n_clusters=4, random_state=42, n_init=10)
km_labels = km.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=km_labels, cmap="viridis")
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            marker="X", s=180, c="red")
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=4, linkage="ward")
hc_labels = hc.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=hc_labels, cmap="plasma")
plt.title("Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print("K-Means Silhouette Score:",
      round(silhouette_score(X, km_labels), 3))
print("Hierarchical Silhouette Score:",
      round(silhouette_score(X, hc_labels), 3))
