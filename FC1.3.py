# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 02:02:21 2019

@author: Chenghai Li
"""
import math
import time
import random
import numpy as np
from scipy.integrate import odeint
from numpy import random as rd
from matplotlib import pyplot as plt

dtype = np.float32

class grad_node():
    
    def __init__(self, value):
         
        self.value = value
        self.grad = None
        self.accumulate_grad = None
        self.update_grad = None
        self.parent = None
        self.child = None
        self.transpose = False
        self.update_pack = None
         
    def reset(self):
        
        self.grad = None
        self.accumulate_grad = None
        self.update_grad = None
        self.parent = None
        self.child = None
        
   
def add_matrix(a, b):
    
    return a+b

def dot_matrix(a, b):
    
    return np.dot(a, b)  
     
class dense_connect():
    
    def __init__(self, dim_in, dim_out):
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = grad_node( np.array(rd.rand(dim_out, dim_in)-0.5, dtype = dtype) )
        self.bias = grad_node( np.array(rd.rand(dim_out, 1)-0.5, dtype = dtype) )
       

    def forward(self, x):
        
        y = grad_node( add_matrix( dot_matrix( self.weight.value, x.value ), self.bias.value) )
        y.child = [x, self.weight, self.bias]
        
        x.parent = [y]
        x.grad = self.weight.value
        x.transpose = True
        
        self.weight.parent = [y]
        self.weight.grad = x.value.transpose()
        
        self.bias.parent = [y]
        #self.bias.grad = np.eye(self.dim_out, dtype = float)
        self.bias.grad = np.eye(1, dtype = dtype)
        
        return y
    
class activation():
   
    def forward(x):
        
        y = grad_node( np.tanh(x.value) )
        y.child = [x]
    
        x.parent = [y]
        x.grad = np.diagflat( (1 - np.tanh(x.value) ** 2).reshape([-1]) )
        x.transpose = True
        return y
    
class loss():

    def forward(x, y):
        
        L = grad_node( np.sqrt( ( (x.value - y.value) ** 2 ).sum()) / x.value.size )
        L.accumulate_grad = 1
        L.child = [x]
        x.parent = [L]
        x.grad = (1/L.value * (x.value - y.value) / x.value.size)

        return L
    
FC1 = dense_connect(2,25)
FC2 = dense_connect(25,25)
FC3 = dense_connect(25,2)

seq = [FC1, activation, FC2, activation, FC3]

def forward(x, y, seq):
    
    for op in seq:
        
        #print(x.value)
        #print()
        x = op.forward(x)
        
    #print(x.value)
    L = loss.forward(x, y)

    return L

def predict(x, seq):
    
    for op in seq:
        
        #print(x.value)
        #print()
        x = op.forward(x)
        
    return x.value.reshape([-1])

def back(node, accumulate_grad): #先序遍历
    

    if node.transpose == False:
        accumulate_grad = dot_matrix( accumulate_grad, node.grad)
    else:
        accumulate_grad = dot_matrix( node.grad.transpose(), accumulate_grad) 
  
    if type(node.update_grad) == type(None):
        node.update_grad = accumulate_grad
    else:
        node.update_grad += accumulate_grad
        
    if node.child==None:
        return
    
    for childs in node.child:
        back(childs, accumulate_grad)

    
def update(node_list, lr):
    
    for nodes in node_list:
    
        nodes.value -= lr * nodes.update_grad
        
class Adam():
    
    def __init__(self, lr):
        
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-6
        
    def update(self, node_list):
        
        global update_value
        
        for i in range (len(node_list)):
            nodes = node_list[i]
            if type(nodes.update_pack) == type(None):
                m = (1 - self.beta1) * nodes.update_grad
                v = (1 - self.beta2) * (nodes.update_grad ** 2)
                nodes.update_pack = [m, v]
                nodes.value -= (self.lr * m) / (np.sqrt(v) + self.eps)
            else:
                m ,v = nodes.update_pack
                m = self.beta1 * m + (1 - self.beta1) * nodes.update_grad
                v = self.beta2 * v + (1 - self.beta2) * (nodes.update_grad ** 2)
                nodes.update_pack = [m, v]
                nodes.value -= (self.lr * m) / (np.sqrt(v) + self.eps)
                update_value[i].append(((self.lr * m) / (np.sqrt(v) + self.eps)).mean())

        
def grad_zero(node_list):
    
    for nodes in node_list:
    
        nodes.reset()
        
        
def lr_function(x):
    
    return 0.0005*math.e**(-x/50000)     
        
def train(data, batch, epoch):
    
    node_list = [FC1.bias, FC1.weight, FC2.bias, FC2.weight, FC3.bias, FC3.weight]
    loss_list = []
    
    optimizer = Adam(0.001)
    
    for i in range (epoch):
        
        temp = data.get(batch)
        for obj in temp:
            
            x = grad_node(obj[0])
            y = grad_node(obj[1])
            
            loss_f = forward(x, y, seq)
            loss_list.append(loss_f.value)
            #print('Epochs:',i,' loss:',loss_f.value)
            back(loss_f.child[0], np.eye(2, dtype = dtype))
 
        optimizer.update(node_list)
        grad_zero(node_list)
    
    return loss_list

class dataset():   
    
    def __init__(self, x, y):
        
        self.x = x
        self.y = y
        self.len = len(x)
        
    def get(self, batch):
        
        ids = random.sample(range(0, self.len), batch)
        return [[self.x[i], self.y[i]] for i in ids]
        
   
x_in = np.load('input_train.npy').reshape([-1,2,1]).astype(dtype)
y_out = np.load('output_train.npy').reshape([-1,2,1]).astype(dtype)

data = dataset(x_in, y_out)
    
update_value = [[],[],[],[],[],[]]
start = time.time()

loss_list = train(data, 32, 50000)

end = time.time()


    
plt.plot(update_value[4])

print(end-start)

loss_list = np.array(loss_list)
loss_list = np.log(loss_list)
#plt.plot(loss_list)

def func(w, t):

    x, y = w
    return np.array([x-4*y, 4*x-7*y])

   
a = [0, -1]
output = []
ori = np.array(a).astype(dtype)
output.append(ori)

for i in range (20):
    change = predict(grad_node(ori.reshape([2,1])),seq)
    ori = change + ori
    output.append(ori)

x0 = []
y0 = []

fig = plt.figure()
t1 = np.arange(0, 2, 0.001) 

track1 = odeint(func, (0, -1), t1)
    
for i in range (21):
    x0.append(output[i][0])
    y0.append(output[i][1])

t2 = np.arange(0, 2.1, 0.1) 

plt.subplot(1,3,1)
plt.plot(t1,track1[:,0],c='r')
plt.scatter(t2, x0, s=15)

plt.subplot(1,3,2)
plt.plot(t1,track1[:,1],c='r')
plt.scatter(t2, y0, s=15)

plt.subplot(1,3,3)
plt.plot(track1[:,0],track1[:,1],c='r')
plt.scatter(x0, y0, s=15)

