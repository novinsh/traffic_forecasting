#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


def generate_dataset(num_points, num_features, dataset_type, heterogenous=False):
    """ 
    Generates 1D or 2D regression datasets with a simple linear or non-linear 
    relationship between the input and output with a homogenous or heterogenous
    white noise.
    """
    # Set random seed for reproducibility
    np.random.seed(0)

    X = np.random.uniform(-5, 5, num_points).reshape(-1, 1)
    noise = np.random.normal(0, 0.5 + np.abs(X) if heterogenous else np.ones(num_points)).squeeze()

    # 1D data
    if num_features == 1 and dataset_type == 'linear': 
        y = (2 * X.squeeze() + 3) + noise
        return X, y
    elif num_features == 1 and dataset_type == 'non-linear': 
        y = X.squeeze()**2 - 2*X.squeeze() + 3 + noise 
        return X, y

    # 2D data
    elif num_features == 2 and dataset_type == 'linear': 
        X1 = np.random.uniform(-5, 5, num_points).reshape(-1, 1)
        X2 = np.random.uniform(-5, 5, num_points).reshape(-1, 1)
        X = np.column_stack((X1, X2))  # Combine X1 and X2 into a single input array

        noise = np.random.normal(0, 0.5 + np.abs(X1)+np.abs(X2) if heterogenous else np.ones(num_points)).reshape(-1,1)

        y = X1*2 + X2*3 + 5 + noise
        return X, y

    elif num_features == 2 and dataset_type == 'non-linear': 
        X1 = np.random.uniform(-5, 5, num_points).reshape(-1, 1)
        X2 = np.random.uniform(-5, 5, num_points).reshape(-1, 1)
        X = np.column_stack((X1, X2))  # Combine X1 and X2 into a single input array

        noise = np.random.normal(0, 0.5 + np.abs(X1)+np.abs(X2) if heterogenous else np.ones(num_points)).reshape(-1,1)

        y = X1**2 - 2*X2 + 3 + noise
        return X, y
    else:
        raise ValueError("permitted num_features: 1 or 2. \n \
                         permitted dataset_types: 'linear' or 'non-linear'.")


def generate_count_dataset(n_samples, num_features, dataset_type, heterogenous=False):
    """ 
    Generates 1D or 2D count datasets with a simple linear or non-linear 
    relationship between the input and output with a homogenous or heterogenous
    white noise.
    """
    # TODO: make it possible to have the poisson parameter be heterogenous!
    if num_features == 1 and dataset_type == 'linear':
        X = np.random.uniform(low=-5, high=5, size=n_samples).reshape(-1,1)

        # Generate homogenous noise 
        # TODO: make it possible to have heterogenous noise
        noise = np.random.normal(loc=0, scale=1, size=n_samples)

        # Generate output data with a quadratic relationship and heteroscedastic noise
        # y = a*X + b + ε
        a = 0.5
        b = 1
        # Apply the exponential function to ensure non-negative parameter for the Poisson distribution
        poisson_param = np.exp(a*X.squeeze() + b)
        # Introduce Poisson distribution to generate integer count values
        y = np.random.poisson(poisson_param) + noise
        return X, y[:,np.newaxis]
    elif num_features == 1 and dataset_type == 'non-linear':
        X = np.random.uniform(low=-5, high=5, size=n_samples).reshape(-1,1)

        # Generate homogenous noise 
        # TODO: make it possible to have heterogenous noise
        noise = np.random.normal(loc=0, scale=1, size=n_samples)

        # Generate output data with a quadratic relationship and heteroscedastic noise
        # y = a*X^2 + b*X + c + ε
        a = 0.05
        b = 0.5
        c = 1
        # Apply the exponential function to ensure non-negative parameter for the Poisson distribution
        poisson_param = np.exp(a*X.squeeze()**2 + b*X.squeeze() + c)
        # Introduce Poisson distribution to generate integer count values
        y = np.random.poisson(poisson_param) + noise
        return X, y[:,np.newaxis]      
    elif num_features == 2 and dataset_type == 'linear':
        X1 = np.random.uniform(low=-5, high=5, size=n_samples)
        X2 = np.random.uniform(low=-5, high=5, size=n_samples)
        X = np.column_stack((X1, X2))  # Combine X1 and X2 into a single input array

        # Generate noise with heteroscedasticity and varying mean
        # TODO: make it possible to have heterogenous noise 
        epsilon_mean = 0 #np.cos(X[:,0]) + np.sin(X[:,1])
        epsilon_std = 1 #+ np.abs(X[:,0])/5  # Varying standard deviation
        epsilon = np.random.normal(loc=epsilon_mean, scale=epsilon_std, size=n_samples)
        epsilon = np.random.normal(loc=epsilon_mean, scale=epsilon_std, size=n_samples)

        # Generate output data with a quadratic relationship and heteroscedastic noise
        # y = a*X1^2 + b*X2^2 + c*X1*X2 + d*X1 + e*X2 + f + ε
        a = 0.2
        b = 0.15
        c = 3
        # Apply the exponential function to ensure non-negative parameter for the Poisson distribution
        lam = a*X[:,0] + b*X[:,1] + c
        poisson_param = np.exp(lam)  
        y = np.random.poisson(poisson_param) + epsilon
        return X, y[:,np.newaxis]
    elif num_features == 2 and dataset_type == 'non-linear':
        print("ASDASDADASDASD")
        X1 = np.random.uniform(low=-5, high=5, size=n_samples)
        X2 = np.random.uniform(low=-5, high=5, size=n_samples)
        X = np.column_stack((X1, X2))  # Combine X1 and X2 into a single input array

        # Generate noise with heteroscedasticity and varying mean
        # TODO: make it possible to have heterogenous noise 
        epsilon_mean = 0 #np.cos(X[:,0]) + np.sin(X[:,1])
        epsilon_std = 1 #+ np.abs(X[:,0])/5  # Varying standard deviation
        epsilon = np.random.normal(loc=epsilon_mean, scale=epsilon_std, size=n_samples)
        epsilon = np.random.normal(loc=epsilon_mean, scale=epsilon_std, size=n_samples)

        # Generate output data with a quadratic relationship and heteroscedastic noise
        # y = a*X1^2 + b*X2^2 + c*X1*X2 + d*X1 + e*X2 + f + ε
        a = 0.05
        b = 0.08
        c = -0.03
        d = 0.2
        e = 0.15
        f = 3
        # Apply the exponential function to ensure non-negative parameter for the Poisson distribution
        lam = a*X[:,0]**2 + b*X[:,1]**2 + c*X[:,0]*X[:,1] + d*X[:,0] + e*X[:,1] + f
        poisson_param = np.exp(lam)  
        y = np.random.poisson(poisson_param) #+ epsilon
        return X, y[:,np.newaxis]
    else:
        raise ValueError("permitted num_features: 1 or 2. \n \
                         permitted dataset_types: 'linear' or 'non-linear'.")


def plot_dataset(X, y, y_pred=None, y_pred_plane_call_back=None, title=''):
    if X.shape[1] > 1:
        fig, axes = plt.subplot_mosaic("AB",
                                    per_subplot_kw={('A'): {'projection': '3d'}},
                                    gridspec_kw={'width_ratios': [1, 1],
                                                'wspace': 0.5, 'hspace': 0.1},
                                    figsize=(10, 5))
        axes['A'].set_title(title)
        axes['A'].scatter(X[:,0], X[:,1], y, alpha=0.5, label='observation')
        axes['A'].set_xlabel('X1')
        axes['A'].set_ylabel('X2')
        axes['A'].set_zlabel('Y')
        axes['B'].hist(y, label='observation', alpha=0.7)
        if y_pred_plane_call_back is not None:
            X1_plane, X2_plane, Y_plane = y_pred_plane_call_back()
            # Plot the model output as a plane
            axes['A'].plot_surface(X1_plane, X2_plane, Y_plane, alpha=0.5, color='orange', label='prediction')
            # axes['A'].legend()

        if y_pred is not None:
            # axes['B'].set_title(f"MSE: {mean_squared_error(y, y_pred):2.2f}")
            axes['B'].hist(y_pred, label='prediction', alpha=0.7)
            axes['B'].legend()
        plt.legend()
        plt.show()
    else:
        fig, axes = plt.subplots(1,2,figsize=(10,5))
        axes[0].set_title(title)
        axes[0].scatter(X, y, label='observation')
        if y_pred is not None:
            idx = np.argsort(X, axis=0)[:,0]
            # axes[0].plot(X[idx], y_pred[idx], color='orange', linewidth=3, label='prediction')
            axes[0].scatter(X, y_pred, label='prediction')
            axes[1].hist(y_pred, alpha=0.7, label='prediction')
        axes[0].set_xlabel('Input')
        axes[0].set_ylabel('Output')
        axes[0].grid(True, alpha=0.5)
        axes[1].hist(y, alpha=0.7, label='observation')
        axes[1].legend()
        axes[0].legend()
        plt.show()


def fit_polynomial_model(X, y, family=sm.families.Poisson()):
    """ 
    A simple family of models that cover the data generating process.
    """
    # Define polynomial features
    poly = PolynomialFeatures(degree=2)  # Adjust the degree as needed
    # Transform the input features to polynomial features
    X_poly = poly.fit_transform(X)
    # Fit the Poisson regression model
    model = sm.GLM(y, X_poly, family=family).fit()
    # Predict the output
    y_pred = model.predict(X_poly)

    def y_pred_plane():
        # Create a meshgrid for plotting the plane
        x1_range = np.linspace(-5, 5, 50)
        x2_range = np.linspace(-5, 5, 50)
        X1_plane, X2_plane = np.meshgrid(x1_range, x2_range)
        Y_plane = model.predict(poly.transform(np.column_stack((X1_plane.ravel(), X2_plane.ravel()))))
        Y_plane = Y_plane.reshape(X1_plane.shape)
        return X1_plane, X2_plane, Y_plane

    return model, poly, y_pred, y_pred_plane


# Gaussian data
#%%
%matplotlib inline
# 1D data
X, y = generate_dataset(num_points=1000, 
                        num_features=1, 
                        dataset_type='linear', 
                        heterogenous=False)
# plot_dataset(X, y, title='Linear+Normal+Homogenous')
model, poly, y_pred, y_pred_plane = fit_polynomial_model(X, y, family=sm.families.Gaussian())
plot_dataset(X, y, y_pred, y_pred_plane, title='Linear+Normal+Homogenous')


X, y = generate_dataset(num_points=1000, 
                        num_features=1, 
                        dataset_type='non-linear', 
                        heterogenous=False)
# plot_dataset(X, y, title='Nonlinear+Normal+Homogenous')
model, poly, y_pred, y_pred_plane = fit_polynomial_model(X, y, family=sm.families.Gaussian())
plot_dataset(X, y, y_pred, y_pred_plane, title='Nonlinear+Normal+Homogenous')


X, y = generate_dataset(num_points=1000, 
                        num_features=1, 
                        dataset_type='linear', 
                        heterogenous=True)
# plot_dataset(X, y, title='Linear+Normal+Heterogenous')
model, poly, y_pred, y_pred_plane = fit_polynomial_model(X, y, family=sm.families.Gaussian())
plot_dataset(X, y, y_pred, y_pred_plane, title='Linear+Normal+Heterogenous')


X, y = generate_dataset(num_points=1000, 
                        num_features=1, 
                        dataset_type='non-linear', 
                        heterogenous=True)
# plot_dataset(X, y, title='Nonlinear+Normal+Heterogenous')
model, poly, y_pred, y_pred_plane = fit_polynomial_model(X, y, family=sm.families.Gaussian())
plot_dataset(X, y, y_pred, y_pred_plane, title='Nonlinear+Normal+Heterogenous')

#%% 
# 2D data
# %matplotlib qt
%matplotlib inline
X, y = generate_dataset(num_points=1000, 
                        num_features=2, 
                        dataset_type='linear', 
                        heterogenous=False)
# plot_dataset(X, y, title='Linear+Normal+Homogenous')
model, poly, y_pred, y_pred_plane = fit_polynomial_model(X, y, family=sm.families.Gaussian())
plot_dataset(X, y, y_pred, y_pred_plane, title='Linear+Normal+Homogenous')


X, y = generate_dataset(num_points=1000, 
                        num_features=2, 
                        dataset_type='non-linear', 
                        heterogenous=False)
# plot_dataset(X, y, title='Nonlinear+Normal+Homogenous')
model, poly, y_pred, y_pred_plane = fit_polynomial_model(X, y, family=sm.families.Gaussian())
plot_dataset(X, y, y_pred, y_pred_plane, title='Nonlinear+Normal+Homogenous')


X, y = generate_dataset(num_points=1000, 
                        num_features=2, 
                        dataset_type='linear', 
                        heterogenous=True)
# plot_dataset(X, y, title='Linear+Normal+Heterogenous')
model, poly, y_pred, y_pred_plane = fit_polynomial_model(X, y, family=sm.families.Gaussian())
plot_dataset(X, y, y_pred, y_pred_plane, title='Linear+Normal+Heterogenous')


X, y = generate_dataset(num_points=1000, 
                        num_features=2, 
                        dataset_type='non-linear', 
                        heterogenous=True)
# plot_dataset(X, y, title='Nonlinear+Normal+Heterogenous')
model, poly, y_pred, y_pred_plane = fit_polynomial_model(X, y, family=sm.families.Gaussian())
plot_dataset(X, y, y_pred, y_pred_plane, title='Nonlinear+Normal+Heterogenous')

#%% 

# Count data (poisson)
# %matplotlib qt
%matplotlib inline
X, y = generate_count_dataset(n_samples=1000, 
                            num_features=1, 
                            dataset_type='linear', 
                            heterogenous=False)
# plot_dataset(X, y, title='')
model, poly, y_pred, y_pred_plane = fit_polynomial_model(X, y, family=sm.families.Poisson())
plot_dataset(X, y, y_pred, y_pred_plane)

X, y = generate_count_dataset(n_samples=1000, 
                            num_features=1, 
                            dataset_type='non-linear', 
                            heterogenous=False)
# plot_dataset(X, y, title='')
model, poly, y_pred, y_pred_plane = fit_polynomial_model(X, y, family=sm.families.Poisson())
plot_dataset(X, y, y_pred, y_pred_plane)

X, y = generate_count_dataset(n_samples=1000, 
                            num_features=2, 
                            dataset_type='linear', 
                            heterogenous=False)
# plot_dataset(X, y, title='')
model, poly, y_pred, y_pred_plane = fit_polynomial_model(X, y, family=sm.families.Poisson())
plot_dataset(X, y, y_pred, y_pred_plane)

X, y = generate_count_dataset(n_samples=1000, 
                            num_features=2, 
                            dataset_type='non-linear', 
                            heterogenous=False)
# plot_dataset(X, y, title='')
model, poly, y_pred, y_pred_plane = fit_polynomial_model(X, y, family=sm.families.Poisson())
plot_dataset(X, y, y_pred, y_pred_plane)


#%%
X, y = generate_count_dataset(n_samples=2000, 
                            num_features=2, 
                            dataset_type='non-linear', 
                            heterogenous=False)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.8)

model, poly, y_tr_pred, y_pred_plane = fit_polynomial_model(X_tr, y_tr, family=sm.families.Poisson())
y_te_pred = model.predict(poly.transform(X_te))

mse_tr = mean_squared_error(y_tr, y_tr_pred)
mse_te = mean_squared_error(y_te, y_te_pred)

%matplotlib qt
plot_dataset(X_tr, y_tr, y_tr_pred, y_pred_plane, title=f"train - MSE: {mse_tr:2.2f}")
plot_dataset(X_te, y_te, y_te_pred, y_pred_plane, title=f"test - MSE: {mse_te:2.2f}")

#%%
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def fit_nn_model(X, y, tag=''):
    input_shape=X.shape[1]
    output_shape=y.shape[1]
    model = Sequential()
    model.add(Dense(256, input_shape=(input_shape,), activation='relu', kernel_regularizer=regularizers.L2(1e-4)))
    model.add(Dense(128, input_shape=(input_shape,), activation='relu', kernel_regularizer=regularizers.L2(1e-4)))
    model.add(Dense(64, input_shape=(input_shape,), activation='relu', kernel_regularizer=regularizers.L2(1e-4)))
    model.add(Dense(output_shape, activation='softplus'))

    adam = Adam(learning_rate=0.005)

    file_path = f'tmp/ckpt/model-lowest-error_{tag}.h5' 
    checkpoint = ModelCheckpoint(file_path, 
        verbose=1, 
        monitor='val_loss',
        save_best_only=True, 
        mode='min'
    )  
    model.compile(loss='poisson', optimizer=adam, metrics=['mean_squared_error'])
    # model.compile(loss='mean_squared_error', optimizer=adam)
    model.fit(X, y, epochs=50, batch_size=32, verbose=1, validation_split=0.1, callbacks=[checkpoint])
    model = load_model(file_path)

    def y_pred_plane():
        # Create a meshgrid for plotting the plane
        x1_range = np.linspace(-5, 5, 50)
        x2_range = np.linspace(-5, 5, 50)
        X1_plane, X2_plane = np.meshgrid(x1_range, x2_range)
        Y_plane = model.predict(np.column_stack((X1_plane.ravel(), X2_plane.ravel())))
        Y_plane = Y_plane.reshape(X1_plane.shape)
        return X1_plane, X2_plane, Y_plane

    y_tr_pred = model.predict(X_tr)

    return model, y_tr_pred, y_pred_plane

model, y_tr_pred, y_pred_plane = fit_nn_model(X_tr, y_tr)

#%%
%matplotlib qt
y_te_pred = model.predict(X_te)

mse_tr = mean_squared_error(y_tr, y_tr_pred)
mse_te = mean_squared_error(y_te, y_te_pred)

plot_dataset(X_tr, y_tr, y_tr_pred, y_pred_plane, title=f"train - MSE: {mse_tr:2.2f}")
plot_dataset(X_te, y_te, y_te_pred, y_pred_plane, title=f"test - MSE: {mse_te:2.2f}")