import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



traffic = pd.read_csv('TrafficData.csv')
labelenconder = LabelEncoder()
traffic = traffic.drop(['Date','id','source','destination','time'],axis=1)
traffic['factor'] = labelenconder.fit_transform(traffic['factor'])
data = traffic.groupby('factor')['delay_in_min','distance(km)'].mean()
kmeans = KMeans(n_clusters=3)
res = kmeans.fit_predict(data)
print(res)
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.bar(res['factor'].astype(int),res['delay_in_min'].astype(int))
#plt.show()