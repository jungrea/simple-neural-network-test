# -*- coding: utf-8 -*-
"""
@author: chen

"""
import numpy as np
import pickle
import sys
from scipy import misc

def read_one_image(filename):
    im = misc.imread(filename)
    im = im.reshape(1,784)
    return im 
    
def sigmoid(inputs):
    row,col = inputs.shape
    for i in range(row):
        for j in range(col):
            inputs[i,j] = 1.0 / (1.0 + np.exp(- inputs[i,j]))
    return inputs
    
class nn_setup():
    def __init__(self,net,learningRate = 2, epochs = 100, batch = 100, dropoutFraction = 0.05):
        self.net = net
        self.size = net.size
        self.learningRate = learningRate
        self.dropoutFraction = dropoutFraction
        self.epochs = epochs
        self.batch = batch
        self.W = list()
        self.a = list()
        self.d = list()
        self.dW = list()
        self.dropoutMask = list()
        self.L = 0
        
        for i in range(1,self.size):
            weight = (np.random.rand(self.net[i], self.net[i - 1]+1) - 0.5) * 2 * 4 * np.sqrt(6 / (self.net[i] + self.net[i - 1]))
            self.W.append(weight)
            
            weight = np.zeros([self.net[i], self.net[i - 1]+1])
            self.dW.append(weight)

        for i in range(self.size):
            if i == self.size-1:
                a_weight = np.zeros([self.batch, self.net[i]])
            else:      
                a_weight = np.zeros([self.batch, self.net[i]+1])
            self.a.append(a_weight)
        
        if self.dropoutFraction > 0:
            for i in range(self.size):
                if i == self.size-1:
                    dropout_weight = np.zeros([self.batch, self.net[i]])
                else:      
                    dropout_weight = np.zeros([self.batch, self.net[i]+1])
                self.dropoutMask.append(dropout_weight)
            
        for i in range(self.size):
            if i == self.size-1:
                d_weight = np.zeros([self.batch, self.net[i]])
            else:      
                d_weight = np.zeros([self.batch, self.net[i]+1])
            self.d.append(d_weight) 
        
        self.e = np.zeros(self.batch,self.net[self.size - 1])    
        
##----------------------------------------------------------------argv

img = read_one_image(sys.argv[1])
# 加载模型
pkl_file = open('model/model.pkl', 'rb')
nn = pickle.load(pkl_file)
##-------------------前向传播计算输出-------------------
m = img.shape[0]
nn.a[0] = np.hstack((np.ones([m,1]),img))
for i in range(1,nn.size-1):
    nn.a[i] = sigmoid(np.dot(nn.a[i-1],nn.W[i-1].T))
    nn.a[i] = nn.a[i] * (1-nn.dropoutFraction)
    nn.a[i] = np.hstack((np.ones([m,1]),nn.a[i]))
nn.a[nn.size-1] = sigmoid(np.dot(nn.a[nn.size-2],nn.W[nn.size-2].T))
res = nn.a[nn.size-1]
pre_y = np.argmax(res)
        
##-------------------显示结果-------------------
print('\n-----------------------------------------\n',
'the prediction of this number is : ', pre_y,
'\n-----------------------------------------\n')

