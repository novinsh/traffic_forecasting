# This file loads the data and clusters it 
#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn_extra.cluster import KMedoids

# ratiolist = [25,50,75,100]
ratiolist = [25]
n_cluster = 20
for ratio in ratiolist:
    # load data
    df_traffic = pd.read_csv("sample_ratio_" + str(ratio)+"_train.csv", index_col=0)
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

    # cluster the data
    X = values[:,:,1] 
    km = KMedoids(n_clusters=n_cluster, metric='euclidean',method='pam',init='heuristic', max_iter=300, random_state=0)
    km = km.fit(X)
    y_pred = km.predict(X)
    
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
    #save the results for later 
    with open("cluster_model_medoids_ratio_"+str(ratio)+"_train_"+str(n_cluster)+".pkl", "wb") as f:
        pickle.dump(km, f, pickle.HIGHEST_PROTOCOL)

#%%
# Plot two of the clusters members with their medoids 
plt.figure(figsize=(14,4))
for i, yi in enumerate([1,19]): # plot only cluster 1 and 20
    plt.subplot(2, 1, i+1)
    for xx in X[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.05)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, X.shape[1])
    plt.grid(alpha=0.5)
    plt.title(f"Cluster {yi + 1} - #roads: {(y_pred==yi).sum()}")
plt.tight_layout()
plt.savefig("figures/cluster_result.pdf", bbox_inches='tight')
plt.show()
#%%
counts_of_members = []
for yi in range(n_cluster):
    counts_of_members.append((y_pred == yi).sum())

counts_of_members = np.sort(counts_of_members)
# plt.title('Distribution of Cluster Members')
plt.figure(figsize=(7,4))
plt.bar(range(1, len(counts_of_members)+1), counts_of_members)
plt.xticks(range(1, len(counts_of_members)+1), range(1, len(counts_of_members)+1))
plt.xlabel('number of cluster members')
# plt.yscale('log')
plt.grid(alpha=0.4)
plt.savefig("figures/distribution_of_cluster_members.pdf", bbox_inches='tight')
plt.show()