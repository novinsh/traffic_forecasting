# this file loads the trained cluster model and uses its results to train a
# regression that maps from cluster centroids to the traffic count for each edges/roads.
#%%
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

ratio_train = 25
ratio_test = 100
n_cluster = 20

def calculate_MAE(X_val, X_reconstructed):
    """ Calculate the MAE of the predicted matrix compared to the original matrix """
    D = abs(X_val - X_reconstructed)
    MAE = np.mean(np.mean(abs(D)))
    # print("avg error:", MAE)
    max_error = np.max(D)
    perc_99 = np.percentile(D, 99)
    perc_95 = np.percentile(D, 95)
    return MAE, max_error, perc_99, perc_95


def augment_traffic_data(x_train, y_train):
    """ 
    Augment the training data by interpolating data from successive hours
    N is the desired ratio between the size of the resulting dataset and the size of the original dataset
    """
    x_all_data = list()
    y_all_data = list()
    for i in range(-1, len(x_train)-1):
        x_data_1 = x_train[i]
        y_data_1 = y_train[i]
        for j in range(-1, len(x_train)-1):
            x_data_2 = x_train[j]
            x_new_data = x_data_1+ x_data_2
            x_all_data.append(x_new_data)
            y_data_2 = y_train[j]
            y_new_data = y_data_1+ y_data_2
            y_all_data.append(y_new_data)
    return np.array(x_all_data),np.array(y_all_data)

# load train data
df_traffic = pd.read_csv("sample_ratio_"+str(ratio_train)+"_train.csv", index_col=0)
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
with open("cluster_model_medoids_ratio_"+str(ratio_train)+"_train_"+str(n_cluster)+".pkl", "rb") as f:
    km = pickle.load(f)

#%%
# predict clusters 
X = values[:,:,1] 
n_cluster = km.n_clusters
y_pred = km.predict(X)

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

X_ = X[km.medoid_indices_].T
y_ = X.T

print(X_.shape)
print(y_.shape)

X_tr = X_
y_tr = y_

# by default, keras tensors accept float32 and not float64
X_tr = X_tr.astype('float32')
y_tr = y_tr.astype('float32')

#augment data
#X_tr, y_tr = augment_traffic_data(X_tr, y_tr)

print(X_tr.shape)
print(y_tr.shape)

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
from keras import initializers
from keras import regularizers
import keras


# TODO:
# 1. weight initialization - https://keras.io/api/layers/initializers/
# 2. weight regularization - helps to keep epochs high and overfitting minimal: kernel term (w1) / bias term (w0): a w_1 + w_0.  https://keras.io/api/layers/regularizers/
# 3. early stopping - helps to stop training if training already saturates or in other words to detect the overfitting early and stop https://keras.io/api/callbacks/early_stopping/
callback = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=100,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0,
)
model = Sequential()
model.add(Dense(64, input_shape=(input_shape,), activation='relu',
                 #kernel_initializer=initializers.RandomNormal(stddev=0.01),
                 #bias_initializer=initializers.Zeros(),
                 kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                 bias_regularizer=regularizers.L2(1e-4)))
model.add(Dense(128, activation='relu',
                 #kernel_initializer=initializers.RandomNormal(stddev=0.01),
                 #bias_initializer=initializers.Zeros(),
                 kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                 bias_regularizer=regularizers.L2(1e-4)))
model.add(Dense(output_shape, activation='softplus',
                #kernel_initializer=initializers.RandomNormal(stddev=0.01),
                 #bias_initializer=initializers.Zeros())
                 kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
                 #bias_regularizer=regularizers.L2(1e-4))
         
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(X_tr, y_tr, epochs=1000, batch_size=16, verbose=1, validation_split=0.1, callbacks=[callback])

"""
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
"""
# %%

print("train error")
yy_hat = model.predict(X_tr)
MAE, max_error, perc_99, perc_95 = calculate_MAE(y_tr, yy_hat)
print("MAE:", MAE)
print("Max error:", max_error)
print("99th percentile error:", perc_99)
print("95th percentile error:", perc_95)
RMSE = np.sqrt(np.mean(np.power((y_tr - yy_hat), 2)))
print("RMSE:", RMSE)

# %%

# read test data
print("==================================")
print("test error")
df_test = pd.read_csv("sample_ratio_"+str(ratio_test)+"_test.csv", index_col=0)
df_test['edge_id'] = df_test['edge_id'].astype('string')
print(df_test.dtypes)
df_test.head()

# group by edges
ids = []
values = []
timestamps = []
for id, val in df_test.groupby('edge_id'):
    ids.append(id)
    values.append(val)
    timestamps.append(val.index) # since all days have same timestamp, all should be the same

values = np.array(values)
ids = np.array(ids)
timestamps = np.array(timestamps)

print(ids.shape)
print(values.shape)
print(timestamps.shape)

X = values[:,:,1] 
X = np.asarray(X).astype(np.float32)

n_cluster = km.n_clusters

X_te = X[km.medoid_indices_].T
y_te = X.T

print(X_te.shape)
print(y_te.shape)
yy_hat = model.predict(X_te)
MAE, max_error, perc_99, perc_95 = calculate_MAE(y_te, yy_hat)
print("MAE:", MAE)
print("Max error:", max_error)
print("99th percentile error:", perc_99)
print("95th percentile error:", perc_95)
RMSE = np.sqrt(np.mean(np.power((y_te - yy_hat), 2)))
print("RMSE:", RMSE)

# %%
