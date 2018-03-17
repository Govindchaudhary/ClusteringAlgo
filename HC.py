#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
#The linkage criterion determines which distance to use between sets of observation.
dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian distances')
plt.show()

#fitting hierarchial clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X);


#visuliazing the clusters
# X[y_hc==0,0] indicate that we are taking all the values of 1st column of X which belongs to cluster of index 0

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],c='red',s=100,label = 'cluster1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],c='blue',s=100,label = 'cluster2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],c='green',s=100,label = 'cluster3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],c='cyan',s=100,label = 'cluster4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],c='magenta',s=100,label = 'cluster5')


plt.title('cluster of clients')
plt.xlabel('annual income ')
plt.ylabel('Spending score')
plt.show()

