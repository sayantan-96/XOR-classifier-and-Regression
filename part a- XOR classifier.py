#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# In[2]:


#feedforward netword implementation

def init_layers(arch, seed = 123): #use a gaussian kernel init
    np.random.seed(seed)
    number_of_layers = len(arch)
    params_values = {}

    for indx, layer in enumerate(arch):
        layer_indx = indx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        params_values['W' + str(layer_indx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_indx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        
    return params_values

#Activations and their gradients
def tanh(Z):
    c=np.exp(2*Z)
    return (c-1)/(c+1)

def sigmoid(Z):
    return 1./(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def linear(Z):
    return Z

def tanh_backward(dA, Z):
    t = tanh(Z)
    return dA * (1-np.square(t))

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

def linear_backward(dA,Z):
    return np.array(dA)

def sigmoid_backward(dA,Z):
    t=sigmoid(Z)
    return dA*t*(1-t)
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

#single layer forward pass
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    elif activation is "linear":
        activation_func = linear
    else:
        raise Exception('Non-supported activation function')
        
    return activation_func(Z_curr), Z_curr

#Whole forward pass
def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr
        
        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    return A_curr, memory

#Loss- Mean squared error
def mse(Y_hat, Y):
    temp=np.square(np.subtract(Y,Y_hat)).mean()
    return temp/2.

#single layer backprop

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]
    
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    elif activation is "linear":
        backward_activation_func = linear_backward
    else:
        raise Exception('Non-supported activation function')
    
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

#Full backprop

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
    if(np.linalg.norm(Y-Y_hat)<1e-8):
        dA_prev=0
    else:
        dA_prev =  - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat)) #binary cross entropy
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values

#update params
def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        layer_idx=layer_idx + 1
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values
def predict(params_values,nn_arch,X):
    out,_=full_forward_propagation(X, params_values, nn_arch)
    return out
    


# In[3]:


#batching fn
def create_arr(ind,X,Y):
    xb=[]
    yb=[]
    for i in ind:
        xb.append(X[:,i])
        yb.append(Y[:,i])
    return np.array(xb).reshape(X.shape[0],ind.shape[0]),np.array(yb).reshape(Y.shape[0],ind.shape[0])
#Training loop
def train(X, Y, nn_architecture, epochs, learning_rate,K=0):
    params_values = init_layers(nn_architecture)
    cost_history = []
    accuracy_history=[]
    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate*np.exp(-K*i))
        cost = get_cost_value(Y_hat, Y)
        accuracy = get_accuracy_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy_history.append(accuracy)
    return params_values, cost_history,accuracy_history


# In[4]:


#The data
X=np.array(([-1.,-1.],[1.,1.],[-1.,1.],[1.,-1.])).T
Y=np.array([0,0,1.,1.]).reshape(1,-1)
#The architecture
nn_arch=[{"input_dim": 2, "output_dim": 2, "activation": "relu"},
         {"input_dim": 2, "output_dim": 1, "activation": "relu"},
         {"input_dim": 1, "output_dim": 1, "activation": "sigmoid"}
        ]


# In[5]:


params_val,cost_history,accuracy_history=train(X,Y,nn_arch,epochs=200,learning_rate=0.5,K=0)
print("Training done!! final crossentropy loss= ",cost_history[-1])


# In[6]:


#decision surface
x=np.linspace(-1,1,200)
y=np.linspace(-1,1,200)
xp=[]
yp=[]
c=[]
z=[]
gr=[]
for i in x:
    for j in y:
        temp=predict(params_val,nn_arch,np.array(([i,j])).reshape(-1,1))[0,0]
        xp.append(i)
        yp.append(j)
        
        if(temp>0.5):
            c.append('y')
        else:
            c.append('c')
c0=['r','r','b','b']        


# In[23]:


print("Plotting the decision surface!!")
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Loss function during training on left and decision surface on right')
ax1.plot(np.arange(0,len(cost_history)),cost_history,label='training_loss')
ax2.scatter(xp,yp,c=c)
group=Y[0,:]
cdict = {0.: 'r', 1.: 'b'}
for g in np.unique(group):
    ix = np.where(group == g)
    ax2.scatter(X[0,:][ix], X[1,:][ix], c = cdict[g], label = g, s = 100)
ax2.legend()
ax1.set(xlabel='epochs',ylabel="Crossentropy loss")
ax2.set(xlabel='x',ylabel="y")
ax1.grid()
fig.tight_layout()
plt.show()




