import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
iris = pd.read_csv('iris-dataset.csv')

# 
#print(iris.head())
#print(iris.isnull().sum())

# Encode categorical features
label_encoder = LabelEncoder()
label_encoder.fit(iris['species'])
iris['species'] = label_encoder.transform(iris['species'])

#print(iris.head())
#print(label_encoder.classes_)

# Select features for clustering and convert to numpy array
X = iris.drop(columns=['species'])

# Perform K-Means clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)

kmeans.fit(X)

# Get cluster labels
#print(kmeans.labels_)

# Add cluster labels to the original dataset
iris['prediction'] = kmeans.labels_

#print(iris.head(10))
#print()
#print(iris.tail(10))

#iris['prediction'] -= 1
#iris['prediction'] = iris['prediction'].replace({-1: 2})

#correct = 0
#for i in range(150):
#    if iris['species'][i] == iris['prediction'][i]:
#        correct += 1
#print(f"Correctly classified: {correct} out of 150")

# Calculate silhouette score
print("Silhouette Score:")
print(silhouette_score(X, kmeans.labels_))

print("Predicted Labels:")
print(kmeans.labels_)

print("Cluster Centers:")
print(kmeans.cluster_centers_)

print("Confusion Matrix:")
print(confusion_matrix(iris['species'], iris['prediction']))
'''
Confusion Matrix:
[[ 0 50  0]
 [ 3  0 47]
 [36  0 14]]
 This Confusion Matrix indicates that the model is not performing well.
 It also seems that the model (mis-)labeled the '0' as '1', the '1' as '2', and '2' as '0',
 which considering it is unsupervised learning, is not a problem.
 In this case, we will consider it's accuracy as 133 out of 150 (88.67%)
'''

'''# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering of Iris Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.show()
'''

'''# Visualize the clusters with species
plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=iris['species'], cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering of Iris Dataset with Species')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.show()
'''

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centeroids_pca = pca.transform(kmeans.cluster_centers_)

# Visualize the clusters with PCA
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(centeroids_pca[:, 0], centeroids_pca[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering of Iris Dataset with PCA')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.grid()
plt.show()