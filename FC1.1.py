# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:49:42 2019

@author: Chenghai Li
"""

import numpy as np
from numpy import random as rd
from matplotlib import pyplot as plt

class grad_node():
    
     def __init__(self, value):
         
         self.value = value
         self.grad = None
         self.accumulate_grad = None
         self.update_grad = None
         self.parent = None
         self.child = None
         self.transpose = False
         self.require_grad = True
         
class dense_connect():
    
    def __init__(self, dim_in, dim_out):
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = grad_node( rd.rand(dim_out, dim_in)-0.5 )
        self.bias = grad_node( rd.rand(dim_out, 1)-0.5 )
        
    def forward(self, x):
     
        y = grad_node( np.add(self.weight.value.dot(x.value), self.bias.value) )
        y.child = [x, self.weight, self.bias]
        
        x.parent = [y]
        x.grad = self.weight.value
        x.transpose = True
        
        self.weight.parent = [y]
        self.weight.grad = x.value.transpose()
        
        self.bias.parent = [y]
        #self.bias.grad = np.eye(self.dim_out, dtype = float)
        self.bias.grad = np.eye(1, dtype = float)
        
        return y
    
class activation():

    def forward(x):
        
        y = grad_node( np.tanh(x.value) )
        y.child = [x]
    
        x.parent = [y]
        x.grad = np.diag( (1 - np.tanh(x.value) ** 2).flat )
        x.transpose = True
        return y
    
class loss():
    
    def forward(x, y):
        
        L = grad_node( ( (x.value - y.value) ** 2 ).sum() / x.value.size )
        L.accumulate_grad = 1
        L.child = [x]
        x.parent = [L]
        x.grad = (2 * (x.value - y.value) / x.value.size)

        return L
    
FC1 = dense_connect(2,50)
FC2 = dense_connect(50,50)
FC3 = dense_connect(50,2)

seq = [FC1, activation, FC2, activation, FC3]

def forward(x, y, seq):
    
    for op in seq:
        
        #print(x.value)
        #print()
        x = op.forward(x)
        
    #print(x.value)
    L = loss.forward(x, y)

    return L

def back(node, accumulate_grad): #先序遍历
    

    if node.transpose == False:
        accumulate_grad = accumulate_grad.dot(node.grad)
    else:
        accumulate_grad = (node.grad.transpose()).dot(accumulate_grad)
  
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
        
def grad_zero(node_list):
    
    for nodes in node_list:
    
        nodes.update_grad = 0
        
def train(x, y, epoch):
    
    node_list = [FC1.bias, FC1.weight, FC2.bias, FC2.weight, FC3.bias, FC3.weight]
    loss_list = []
    for i in range (epoch):
        
        loss_f = forward(x, y, seq)
        loss_list.append(loss_f.value)
        #print('Epochs:',i,' loss:',loss_f.value)
        back(loss_f.child[0], np.eye(2, dtype = float))
        update(node_list, 0.0001)
        grad_zero(node_list)
    
    return loss_list
        
x = grad_node(np.array([1,2]).reshape([-1,1]))
y = grad_node(np.array([2,3]).reshape([-1,1]))

'''
node_list = [FC1.bias, FC1.weight, FC2.bias, FC2.weight, FC3.bias, FC3.weight]
loss_f = forward(x, y, seq)
back(loss_f.child[0], np.eye(2, dtype = float))
#update(node_list, 0.00001)
'''

loss_list = train(x, y, 1000)
plt.plot(loss_list)
