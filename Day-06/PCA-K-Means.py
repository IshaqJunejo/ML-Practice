# Trying to implement PCA before K-Means Clustering on the Iris dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
iris = pd.read_csv('iris-dataset.csv')

# Encode categorical features
label_encoder = LabelEncoder()
label_encoder.fit(iris['species'])
iris['species'] = label_encoder.transform(iris['species'])

# PCA
X = iris.drop(columns=['species'])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# K-Means Clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)

kmeans.fit(X_pca)

# Accuracy
print("Silhouette Score:")
print(silhouette_score(X, kmeans.labels_))

print("Predicted Labels:")
print(kmeans.labels_)

print("Cluster Centers:")
print(kmeans.cluster_centers_)

print("Confusion Matrix:")
print(confusion_matrix(iris['species'], kmeans.labels_))

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()

## Okay, so performing PCA before K-Means clustering or after, the results are the same.
## The silhouette score is the same, the cluster centers are the same, and the confusion matrix is the same.