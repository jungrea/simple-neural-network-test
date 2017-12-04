# -*- coding: utf-8 -*-
"""
@author: chen

"""
import numpy as np
import struct
import pickle
from datetime import datetime

def read_image(filename):
    binfile = open(filename , 'rb')
    buf = binfile.read()
     
    index = 0
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')
    
    data = np.zeros((numImages,numRows*numColumns))
    for i in range(numImages): 
        im = struct.unpack_from('>784B' ,buf, index)
        index += struct.calcsize('>784B')
         
        im = np.array(im)
        data[i,:] = im
    return data
    
def read_label(filename):
    binfile = open(filename , 'rb')
    buf = binfile.read()
     
    index = 0
    magic, numLabels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
    
    data = np.zeros((numLabels,10))
    for i in range(numLabels): 
        label = struct.unpack_from('>B' ,buf, index)[0]
        
        label = np.array(label)
        data[i,label] = 1

        index += struct.calcsize('>B')
    return data

def sigmoid(inputs):
    row,col = inputs.shape
    for i in range(row):
        for j in range(col):
            inputs[i,j] = 1 / (1 + np.exp(- inputs[i,j]))
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
        
##----------------------------------------------------------------
if __name__ == '__main__':
    filename_traindata = 'MNIST_data/train-images.idx3-ubyte'
    filename_trainlabel = 'MNIST_data/train-labels.idx1-ubyte'
    filename_testdata = 'MNIST_data/t10k-images.idx3-ubyte'
    filename_testlabel = 'MNIST_data/t10k-labels.idx1-ubyte'
    
    test_data = read_image(filename_testdata)/255
    test_label = read_label(filename_testlabel)
    # 加载模型
    pkl_file = open('model/model.pkl', 'rb')
    nn = pickle.load(pkl_file)
    # 记录时间
    time_start = datetime.now()
    ##-------------------前向传播计算输出-------------------
    m = test_data.shape[0]
    nn.a[0] = np.hstack((np.ones([m,1]),test_data))
    for i in range(1,nn.size-1):
        nn.a[i] = sigmoid(np.dot(nn.a[i-1],nn.W[i-1].T))
        nn.a[i] = nn.a[i] * (1-nn.dropoutFraction)
        nn.a[i] = np.hstack((np.ones([m,1]),nn.a[i]))
    nn.a[nn.size-1] = sigmoid(np.dot(nn.a[nn.size-2],nn.W[nn.size-2].T))
    res = nn.a[nn.size-1]
    pre_y = np.zeros(res.shape[0])
    y_label = np.zeros(res.shape[0])
    count = 0
    ##-------------------计算预测准确率-------------------
    for i in range(res.shape[0]):
        pre_y[i] = np.argmax(res[i,:])
        y_label[i] = np.argmax(test_label[i,:])
        if pre_y[i] == y_label[i]:
            count = count + 1
            
    time_end = datetime.now()
    accuracy = count/y_label.size
    ##-------------------显示结果-------------------
    print('-----------------------------------------\n',
    'test accuracy = ', accuracy, '(',count,'/',y_label.size,')',
    '\n All testing time = ', (time_end.minute -time_start.minute)*60 + (time_end.second-time_start.second) + 
    (time_end.microsecond - time_start.microsecond)/1000000, 's',
    '\n-----------------------------------------\n')
