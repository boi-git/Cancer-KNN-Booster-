#%%
import imp
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import   PCA

df = pd.read_csv('data1.csv')

scaler=StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)
print(scaled_data)

pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)


x_pca.shape
 


# %%
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x_pca)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()








# %%

kmeans = KMeans(n_clusters= 2)
label = kmeans.fit_predict(x_pca)
print(label)


centroids = kmeans.cluster_centers_
u_labels = np.unique(label)

for i in u_labels:
    plt.scatter(x_pca[label == i , 0] , x_pca[label == i , 1] , label = i)
plt.legend()
plt.show()

kmeans = KMeans(n_clusters=4)
label = kmeans.fit_predict(x_pca)
print(label)


centroids = kmeans.cluster_centers_
u_labels = np.unique(label)

for i in u_labels:
    plt.scatter(x_pca[label == i , 0] , x_pca[label == i , 1] , label = i)
plt.legend()
plt.show()



# %%
