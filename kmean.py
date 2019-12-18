from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
from tabulate import tabulate



data = pd.read_csv('TrafficData.csv')

#print(data.shape)
#print(data.head())


f1 = data.groupby('factor').mean().delay
f2 = data.groupby('factor').mean().distance


X = np.array(list(zip(f1, f2)))




# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
#print(kmeans)
# Centroid values
centroids = kmeans.cluster_centers_
#print(centroids)
#print(kmeans.score(X))

factors = np.array(data['factor'].drop_duplicates())
colors = ["g.","r.","c.","y."]
count =np.array(data.set_index(["factor","delay"]).count(level="factor").id) 
prob = count/data.count().id

print(tabulate([ [factors[i], labels[i], prob[i]]  for i in range(len(X)) ] ,headers =["Factors" , "Category" , "Probability" ] ) )


#for i in range(len(X)):
#    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)


#plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

#plt.show()

