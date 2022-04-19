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

#single layer forward pass
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    if activation is "relu":
        activation_func = relu
    elif activation is "tanh":
        activation_func = tanh
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
    elif activation is "tanh":
        backward_activation_func = tanh_backward
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
   
    dA_prev =  -(Y-Y_hat) #dL/dy_hat=-(y-y_hat) /2 is done in mse
    
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
def train(X, Y, nn_architecture, epochs, learning_rate,batch_size,K=0):
    params_values = init_layers(nn_architecture)
    cost_history = []
    no_of_batches=np.ceil(X.shape[1]/batch_size)
    for i in range(epochs):
        #do batching
        indx=np.arange(0,X.shape[1])
        np.random.shuffle(indx)
        split_indices=np.array_split(indx,no_of_batches)
        #Run batches inside epoch
        for j in range(len(split_indices)):
            xb,yb=create_arr(split_indices[j],X,Y)
            Y_hat, cashe = full_forward_propagation(xb, params_values, nn_architecture)
            grads_values = full_backward_propagation(Y_hat, yb, cashe, params_values, nn_architecture)
            params_values = update(params_values, grads_values, nn_architecture, learning_rate*np.exp(-K*i))
        cost = mse(Y_hat, yb)
        cost_history.append(cost)
    return params_values, cost_history


# In[4]:


#data load- Uncomment the below if you want to load default is generation
#data=pd.read_csv(os.getcwd()+"\\Data.csv")
#You can also generate data
def generate_data(seed=100):
    np.random.seed(seed)
    X=2.*np.random.rand(50)-1.
    Y=np.sin(2*np.pi*X)+0.3*np.random.randn(50)
    return X,Y
X,Y=generate_data()
X=X.reshape(-1,1)
Y=Y.reshape(-1,1)

print("Data generated successfully!! using same seed to preserve reproducibility")
# In[5]:


#Run for partb)ii
nn_arch1=[{"input_dim": 1, "output_dim": 3, "activation": "relu"},
         {"input_dim": 3, "output_dim": 1, "activation": "tanh"},
         {"input_dim": 1, "output_dim": 1, "activation": "linear"}
        ]
params_val1,cost_history1=train(X.T,Y.T,nn_arch1,epochs=5000,learning_rate=0.08,batch_size=50,K=0.001)

print("Training done with 3 hidden Neurons!!")


#Run the regression for part b)iii
nn_arch2=[{"input_dim": 1, "output_dim": 20, "activation": "relu"},
         {"input_dim": 20, "output_dim": 1, "activation": "tanh"},
         {"input_dim": 1, "output_dim": 1, "activation": "linear"}
        ]
params_val2,cost_history2=train(X.T,Y.T,nn_arch2,epochs=5000,learning_rate=0.06,batch_size=10,K=0)

print("Training done with 20 hidden neurons!!")


y_hat2=predict(params_val2,nn_arch2,X.T)
y_hat=predict(params_val1,nn_arch1,X.T)


# In[10]:


x1,y1= zip(*sorted(zip(X[:,0].tolist(), y_hat[0,:].tolist())))
x2,y2= zip(*sorted(zip(X[:,0].tolist(), y_hat2[0,:].tolist())))
te1=str(np.round(cost_history1[-1],2))
te2=str(np.round(cost_history2[-1],2))

print("Mean squared error for 3 hidden neurons="+te1)
print("Mean squared error for 20 hidden neurons="+te2)
print("Plotting all at once!!")
# In[11]:


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(np.arange(0,len(cost_history1)),cost_history1,label='training_loss')
axs[0, 0].set_title("3 hidden units full batch")
axs[0, 1].plot(x1,y1, 'tab:orange')
axs[0, 1].text( 0,0,'MSE='+te1, fontsize = 9)
axs[0, 1].set_title("Y predicted vs X")
axs[1, 0].plot(np.arange(0,len(cost_history2)),cost_history2, 'tab:green')
axs[1, 0].set_title('20 hidden units mini batch')
axs[1, 1].plot(x2, y2, 'tab:red')
axs[1, 1].text( 0.4,0.5,'MSE='+te2, fontsize = 9)
axs[1, 1].set_title('Y_predicted vs X')

for ax in axs.flat:
    ax.set(xlabel='Epoch', ylabel='loss')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
    ax.grid()
plt.show()






