#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values


#using the elbow method  to find the optimal no of clusters
# x-axis point corresponding to the pont on the curve where graph stops its sudden dec behaviour

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++',random_state=0) # init='k-means++' is to avoid the random initialization trap
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # kmeans.inertia_ finds the sum of square of all the points for each cluster from their corresponding cluster 

plt.plot(range(1,11),wcss)
plt.title('elbow diagram')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

# applying the k-means to our dataset and predict the actual results

kmeans = KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(X)

#visuliazing the clusters
# X[y_kmeans==0,0] indicate that we are taking all the values of 1st column of X which belongs to cluster of index 0

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],c='red',s=100,label = 'cluster1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],c='blue',s=100,label = 'cluster2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],c='green',s=100,label = 'cluster3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],c='cyan',s=100,label = 'cluster4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],c='magenta',s=100,label = 'cluster5')


plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='yellow',s=300,label = 'centroids')

plt.title('cluster of clients')
plt.xlabel('annual income ')
plt.ylabel('Spending score')
plt.show()
   
    