# This file loads the data and clusters it 
#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
from tslearn.clustering import TimeSeriesKMeans

# load data
df_traffic = pd.read_csv("full_aggregated_count.csv", index_col=0)
df_traffic['edge_id'] = df_traffic['edge_id'].astype('string')
print(df_traffic.dtypes)
df_traffic.head()

# group by edges
ids = []
values = []
timestamps = []
for id, val in df_traffic.groupby('edge_id'):
    ids.append(id)
    values.append(val)
    timestamps.append(val.index) # since all days have same timestamp, all should be the same

values = np.array(values)
ids = np.array(ids)
timestamps = np.array(timestamps)

print(ids.shape)
print(values.shape)
print(timestamps.shape)

#%%
# cluster the data
X = values[:,:,1] 
n_cluster = 20
km = TimeSeriesKMeans(n_clusters=n_cluster, metric="dtw")
y_pred = km.fit_predict(X)

#%%
# Plot the clustered time series data
plt.figure(figsize=(20, 2*n_cluster))
for yi in range(n_cluster):
    plt.subplot(n_cluster, 1, 1 + yi)
    for xx in X[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.05)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, X.shape[1])
    # plt.ylim(-4, 4)
    plt.title(f"Cluster {yi + 1} - #roads: {(y_pred==yi).sum()}")
plt.tight_layout()
plt.show()

#%%

# save the results for later 
with open("cluster_model.pkl", "wb") as f:
    pickle.dump(km, f, pickle.HIGHEST_PROTOCOL)
