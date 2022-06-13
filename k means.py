#%%
import imp
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import   PCA

df = pd.read_csv('data1.csv')


#df.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)
#df.diagnosis.unique()
#df['diagnosis'] = df['diagnosis'].apply(lambda val: 1 if val == 'M' else 0)






scaler=StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)
print(scaled_data)

pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)





x_pca.shape
 
kmeans = KMeans(n_clusters= 2)
label = kmeans.fit_predict(x_pca)
print(label)




# %%

centroids = kmeans.cluster_centers_
u_labels = np.unique(label)

for i in u_labels:
    plt.scatter(x_pca[label == i , 0] , x_pca[label == i , 1] , label = i)
plt.legend()
plt.show()


# %%
