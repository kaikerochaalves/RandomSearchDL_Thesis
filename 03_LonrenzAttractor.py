# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 22:28:50 2025

@author: Kaike Sa Teles Rocha Alves
@email: kaikerochaalves@outlook.com or kaike.alves@estudante.ufjf.br
@scholar: https://scholar.google.com/citations?user=T5Bm_G0AAAAJ&hl=pt-BR&oi=ao
@linkedin: https://www.linkedin.com/in/kaikerochaalves/
"""

#-----------------------------------------------------------------------------
# Libraries
#-----------------------------------------------------------------------------

# Import libraries
import math
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterSampler
from scipy.stats import reciprocal

# Neural Network
# Initialize ANN
from keras.models import Sequential
# Layers
from keras.layers import InputLayer, Dense, Dropout, Conv1D, GRU, LSTM, MaxPooling1D, Flatten, SimpleRNN
# Optimizers
from keras.optimizers import SGD, Adam
# Visualization
from keras.utils import plot_model
# Save network using early stopping and checkpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
# Wrapper for the ANN model
from scikeras.wrappers import KerasRegressor

# Feature scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Including to the path another fold
import sys

# Including to the path another fold
sys.path.append(r'Functions')
# Import the serie generator
from LorenzAttractorGenerator import Lorenz

#-----------------------------------------------------------------------------
# Create the required folders
#-----------------------------------------------------------------------------


import os

folders_to_create = ["RandomSearchResults", "Graphics", "ModelArchiteture"]

for folder in folders_to_create:
    os.makedirs(folder, exist_ok=True)


#-----------------------------------------------------------------------------
# Parameters
#-----------------------------------------------------------------------------

# Number of epochs for the random search
rnd_epochs = 100
# Number of pochs for the final model
final_epochs = 100
# Save only the best model
save_best = True
# Patience to stop training for the random search
rnd_patience = 10
# Patience to stop training for the final model
final_patience = 10
# Number of random search iterations
rnd_iterations = 100
# Number of partitions for the cross vali9dation
rnd_cv = 3

#-----------------------------------------------------------------------------
# Generate the time series
#-----------------------------------------------------------------------------

Serie = "Lorenz"

# Input parameters
x0 = 0.
y0 = 1.
z0 = 1.05
sigma = 10
beta = 2.667
rho=28
num_steps = 10000

# Creating the Lorenz Time Series
x, y, z = Lorenz(x0 = x0, y0 = y0, z0 = z0, sigma = sigma, beta = beta, rho = rho, num_steps = num_steps)

# Ploting the graphic
plt.rc('font', size=10)
plt.rc('axes', titlesize=15)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x, y, z, lw = 0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

def Create_lag(data, ncols, lag, lag_output = None):
    X = np.array(data[lag*(ncols-1):].reshape(-1,1))
    for i in range(ncols-2,-1,-1):
        X = np.append(X, data[lag*i:lag*i+X.shape[0]].reshape(-1,1), axis = 1)
    X_new = np.array(X[:,-1].reshape(-1,1))
    for col in range(ncols-2,-1,-1):
        X_new = np.append(X_new, X[:,col].reshape(-1,1), axis=1)
    if lag_output == None:
        return X_new
    else:
        y = np.array(data[lag*(ncols-1)+lag_output:].reshape(-1,1))
        return X_new[:y.shape[0],:], y

# Defining the atributes and the target value
X = np.concatenate([x[:-1].reshape(-1,1), y[:-1].reshape(-1,1), z[:-1].reshape(-1,1)], axis = 1)
y = x[1:].reshape(-1,1)

# Spliting the data into train and test
X_train, X_val, X_test = X[:6000,:], X[6000:8000,:], X[8000:,:]
y_train, y_val, y_test = y[:6000,:], y[6000:8000,:], y[8000:,:]

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Reshape the inputs
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.show()

#-----------------------------------------------------------------------------
# ANN architectures
#-----------------------------------------------------------------------------

# Define the function to create models for the optimization method
def build_MLP(n_hidden:int, n_neurons:int, activation:str, learning_rate:float, input_shape:tuple, optim:str):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation=activation))
    model.add(Dense(1))
    if optim == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    elif optim == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        print("Optimizer not defined.")
    model.compile(loss="mse", optimizer=optimizer)
    return model

def build_CNN(n_hidden:int, n_neurons:int, activation:str, filters:int, kernel_size:int, pool_size:int, learning_rate:float, input_shape:tuple, optim:str):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=activation))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation=activation))
    model.add(Dense(1))
    if optim == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    elif optim == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        print("Optimizer not defined.")
    model.compile(loss="mse", optimizer=optimizer)
    return model

def build_RNN(n_hidden:int, n_neurons:int, activation:str, learning_rate:float, input_shape:tuple, optim:str):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    if n_hidden == 1:
        model.add(SimpleRNN(n_neurons))
    elif n_hidden == 2:
        model.add(SimpleRNN(n_neurons, return_sequences=True))
        model.add(SimpleRNN(n_neurons))
    else:
        model.add(SimpleRNN(n_neurons, return_sequences=True))
        for layer in range(1, n_hidden-1):
            model.add(SimpleRNN(n_neurons, return_sequences=True))
        model.add(SimpleRNN(n_neurons))
    
    model.add(Dense(1))
    if optim == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    elif optim == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        print("Optimizer not defined.")
    model.compile(loss="mse", optimizer=optimizer)
    return model

def build_LSTM(n_neurons:int, n_lstm_hidden:int, neurons_dense:int, dropout_rate:float, n_dense_hidden:int, learning_rate:float, input_shape:tuple, optim:str):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    if n_lstm_hidden == 1:
        model.add(LSTM(n_neurons, return_sequences=False))
    elif n_lstm_hidden == 2:
        model.add(LSTM(n_neurons, return_sequences=True))
        model.add(LSTM(n_neurons, return_sequences=False))
    else:
        model.add(LSTM(n_neurons, return_sequences=True))
        for layer in range(1, n_lstm_hidden-1):
            model.add(LSTM(n_neurons, return_sequences=True))
        model.add(LSTM(n_neurons, return_sequences=False))
    for dense_layer in range(n_dense_hidden):
        model.add(Dense(neurons_dense))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    if optim == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    elif optim == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        print("Optimizer not defined.")
    model.compile(loss="mse", optimizer=optimizer)
    return model

def build_GRU(filters:int, kernel_size:int, strides:int, padding:str, n_neurons:int, n_gru_hidden:int, neurons_dense:int, dropout_rate:float, n_dense_hidden:int, learning_rate:float, input_shape:tuple, optim:str):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding))
    if n_gru_hidden == 1:
        model.add(GRU(n_neurons, return_sequences=False))
    elif n_gru_hidden == 2:
        model.add(GRU(n_neurons, return_sequences=True))
        model.add(GRU(n_neurons, return_sequences=False))
    else:
        #model.add(keras.layers.GRU(n_neurons, return_sequences=True))
        for layer in range(n_gru_hidden-1):
            model.add(GRU(n_neurons, return_sequences=True))
        model.add(GRU(n_neurons, return_sequences=False))
    for dense_layer in range(n_dense_hidden):
        model.add(Dense(neurons_dense))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    if optim == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    elif optim == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        print("Optimizer not defined.")
    model.compile(loss="mse", optimizer=optimizer)
    return model

def build_WaveNet(dilation_rate:tuple, repeat:int, activation:str, learning_rate:float, filters:int, kernel_size:int, padding:str, input_shape:tuple, optim:str):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    for rate in dilation_rate * repeat:
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation, dilation_rate=rate))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size))
    model.add(Flatten())
    model.add(Dense(1))
    if optim == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    elif optim == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    else:
        print("Optimizer not defined.")
    model.compile(loss="mse", optimizer=optimizer)
    return model

#-----------------------------------------------------------------------------
# Dict of models and hyperparameters to try
#-----------------------------------------------------------------------------

models_to_try = {
    'MLP': {
        'model_reg': build_MLP,
        'grid': {
            "n_hidden": [0, 1, 2, 3],
            "n_neurons": list(range(1, 100)),
            "activation": ["elu", "exponential", "gelu", "hard_sigmoid", "linear", "relu", "selu", "sigmoid", "softplus", "softsign", "swish", "tanh"],
            "learning_rate": reciprocal(1e-5,0.5),
            "input_shape": [X_train.shape[1:2]],
            "optim": ["SGD", "Adam"]
            }       
    },
    'CNN': {
        'model_reg': build_CNN,
        'grid': {
            "n_hidden": [0, 1, 2, 3],
            "n_neurons": list(range(1, 100)),
            "activation": ["relu"],
            "filters": [64],
            "kernel_size": [2],
            "pool_size": [2],
            "learning_rate": reciprocal(1e-5,0.5),
            "input_shape": [X_train.shape[1:3]],
            "optim": ["SGD", "Adam"]}
    },
    'RNN': {
        'model_reg': build_RNN,
        'grid': {
            "n_hidden": [1,2,3,4,5],
            "n_neurons": list(range(1, 100)),
            "activation": ["relu"],
            "learning_rate": reciprocal(1e-5,0.5),
            "input_shape": [X_train.shape[1:3]],
            "optim": ["SGD", "Adam"]}
    },
    'LSTM': {
        'model_reg': build_LSTM,
        'grid': {
            "n_neurons": list(range(1, 100)),
            "n_lstm_hidden": [1,2,3,4,5],
            "neurons_dense": list(range(1, 100)),
            "dropout_rate": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "n_dense_hidden": [0,1,2,3,4],
            "learning_rate": reciprocal(1e-5,0.5),
            "input_shape": [X_train.shape[1:3]],
            "optim": ["SGD", "Adam"]}    
    },
    'GRU': {
        'model_reg': build_GRU,
        'grid': {
            "filters": [2,4,8,16,32,64],
            "kernel_size": [1,2,3,4,5],
            "strides": [1,2,3,4,5],
            "padding": ["valid"],
            "n_neurons": list(range(1, 100)),
            "n_gru_hidden": [1,2,3,4,5],
            "neurons_dense": list(range(1, 100)),
            "dropout_rate": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "n_dense_hidden": [0,1,2,3,4],
            "learning_rate": reciprocal(1e-5,0.5),
            "input_shape": [X_train.shape[1:3]],
            "optim": ["SGD", "Adam"]}       
    },
    'WaveNet': {
        'model_reg': build_WaveNet,
        'grid': {
            "dilation_rate": [(1),(1,2),(1,2,4),(1,2,4,8),(1,2,4,8,16),(1,2,4,8,16,32)],
            "repeat": [1,2],
            "activation": ["relu"],
            "learning_rate": reciprocal(1e-5,0.5),
            "filters": [10,20],
            "kernel_size": [1,2,3,4,5],
            "padding": ["valid","causal"],
            "input_shape": [X_train.shape[1:3]],
            "optim": ["SGD", "Adam"]}       
    },
    }

#-----------------------------------------------------------------------------
# Start simulations
#-----------------------------------------------------------------------------

# Store the summary of the errors
results = []

for model_name, config in models_to_try.items():
    
    #-----------------------------------------------------------------------------
    # Perform the random search approach for each model
    #-----------------------------------------------------------------------------
    
    # Create a sampler for 1 iteration
    sampler = ParameterSampler(config["grid"], n_iter=1)
    
    # Get the dictionary from the sampler
    instance_dict = list(sampler)[0]
    
    # Wrapper around keras model built
    keras = KerasRegressor(model=config["model_reg"], **instance_dict)
    
    # Set the model
    rnd_search_cv = RandomizedSearchCV(estimator=keras, param_distributions=config["grid"], n_iter=rnd_iterations, cv = rnd_cv)
    
    # Run the random search
    rnd_search_cv.fit(X_train, y_train, epochs = rnd_epochs, validation_data=(X_val, y_val), callbacks=[EarlyStopping(patience=rnd_patience)])

    # Print the best model parameters
    print(f'\nBest parameters:\n {rnd_search_cv.best_params_}')
    
    # Print the best model score
    print(f'\nBest score:\n {rnd_search_cv.best_score_}\n\n')
    
    # Implement the prediction method
    y_pred = rnd_search_cv.predict(X_test)
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE:", RMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(y_test.flatten())
    print("NDEI:", NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test, y_pred)
    print("MAE:", MAE)
    
    # Plot the graphic
    plt.figure(figsize=(19.20,10.80))
    plt.rc('font', size=30)
    plt.rc('axes', titlesize=30)
    plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
    plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
    plt.ylabel('Output')
    plt.xlabel('Samples')
    plt.legend(loc='upper left')
    plt.savefig(f'Graphics/{model_name}_{Serie}.eps', format='eps', dpi=1200)
    plt.show()
    
    # -----------------------------------------------------------------------------
    # Run and save the network for the best hyperparameters
    # -----------------------------------------------------------------------------
    
    # Define the neural network
    model = config["model_reg"](**rnd_search_cv.best_params_)
    
    # Checkpoint functions to recover the best model
    checkpoint_cb = ModelCheckpoint(f'RandomSearchResults/{model_name}_{Serie}.keras', save_best_only=save_best)
    early_stopping_cb = EarlyStopping(patience=final_patience,restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs = final_epochs, validation_data=(X_val, y_val), callbacks=[checkpoint_cb, early_stopping_cb])
    
    # Implement the prediction method
    y_pred = model.predict(X_test)
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE:", RMSE)
    # Compute the Normalized Root Mean Square Error
    NRMSE = RMSE/(y_test.max() - y_test.min())
    print("NRMSE:", NRMSE)
    # Compute the Non-Dimensional Error Index
    NDEI= RMSE/st.stdev(y_test.flatten())
    print("NDEI:", NDEI)
    # Compute the Mean Absolute Error
    MAE = mean_absolute_error(y_test, y_pred)
    print("MAE:", MAE)
    # Compute the Mean Absolute Percentage Error
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    print("MAPE:", MAPE)
    
    # Include the results in a list
    result = f'{model_name} & {NRMSE:.5f} & {NDEI:.5f} & {MAPE*100:.2f} & -'
    results.append(result)
    print(f"\n{result}")
    
    # Plot the graphic
    plt.figure(figsize=(19.20,10.80))
    plt.rc('font', size=30)
    plt.rc('axes', titlesize=30)
    plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
    plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
    plt.ylabel('Output')
    plt.xlabel('Samples')
    plt.legend(loc='upper left')
    plt.savefig(f'Graphics/{model_name}_{Serie}.eps', format='eps', dpi=1200)
    plt.show()
    
    # Print the summary of the model
    print(model.summary())
    
    # Plot the model architeture
    # You must install pydot (`pip install pydot`) and install graphviz (https://graphviz.gitlab.io/download/).
    plot_model(model, to_file=f'ModelArchiteture/{model_name}_{Serie}.png', show_shapes=True, show_layer_names=True)
    
# Print the results
for r in results:
    print(r)