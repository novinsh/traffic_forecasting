# this file loads the trained cluster model and uses its results to train a
# regression that maps from cluster centroids to the traffic count for each edges/roads.
#%%
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

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

# load the cluster model
with open("cluster_model.pkl", "rb") as f:
    km = pickle.load(f)

#%%
# predict clusters 
X = values[:,:,1] 
n_cluster = km.n_clusters
y_pred = km.predict(X[:,:,np.newaxis])

#%%
# Plot the clustered time series data
plt.figure(figsize=(n_cluster, 2*n_cluster))
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
from sklearn.model_selection import train_test_split

# transpose so that n_clusters will be n_features

cluster_no=0
ts_in_cluster_zero = values[y_pred == cluster_no]
#X_ = np.random.choice(ts_in_cluster_zero, size=10)


X_ = km.cluster_centers_[:,:,0].T
y_ = X.T

print(X_.shape)
print(y_.shape)

X_tr, X_te, y_tr, y_te = train_test_split(X_, y_, train_size=0.9)

# by default, keras tensors accept float32 and not float64
X_tr = X_tr.astype('float32')
X_te = X_te.astype('float32')
y_tr = y_tr.astype('float32')
y_te = y_te.astype('float32')

print(X_tr.shape)
print(y_tr.shape)
print(X_te.shape)
print(y_te.shape)

input_shape = X_tr.shape[1]  # network's input has same dimensionality as the number of clusters
output_shape = y_tr.shape[1] # network's output has as many heads as the number of edges/roads in our dataset

# print out model's input-output shapes
print(f"Input shape: {input_shape}")
print(f"Output shape: {output_shape}")
#%%
# simple MLP 
# mapping centroids to edges!
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

model = Sequential()
model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(output_shape, activation='softplus'))

model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(X_tr, y_tr, epochs=100, batch_size=16, verbose=1, validation_split=0.1)

y_hat = model.predict(X_te)

if (y_hat < 0).sum():
    print(f"Negative values: {(y_hat < 0).sum()}")
    print(f"Average magnitude of negative values: {y_hat[y_hat < 0].mean():2.4f}")
    y_hat = np.clip(y_hat, a_min=0, a_max=None) # hack. Necessary when output activation is something like linear!

mae = mean_absolute_error(y_te, y_hat)
mse = mean_squared_error(y_te, y_hat)

print(f"MAE: {mae}") 
print(f"RMSE: {np.sqrt(mse)}")

# histogram of the errors 
plt.hist(y_te.flatten(), alpha=0.5, label='obs',bins=100)
plt.hist(y_hat.flatten(), alpha=0.5, label='pred',bins=100)
plt.legend()
plt.show()
# %%
yy_hat = model.predict(X_tr)
mae_ = mean_absolute_error(y_tr, yy_hat)
mse_ = mean_squared_error(y_tr, yy_hat)

print(f"MAE: {mae_}") 
print(f"RMSE: {np.sqrt(mse_)}")
# %%
