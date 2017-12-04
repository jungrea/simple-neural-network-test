# -*- coding: utf-8 -*-
"""
@author: chen

"""
import numpy as np
import struct
import os
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
 
                         
def sigmoid(inputs):
    row,col = inputs.shape
    for i in range(row):
        for j in range(col):
            inputs[i,j] = 1 / (1 + np.exp(- inputs[i,j]))
    return inputs
    
##----------------------------------------------------------------
if __name__ == '__main__':
    # 数据库文件夹选择
    filename_traindata = 'MNIST_data/train-images.idx3-ubyte'
    filename_trainlabel = 'MNIST_data/train-labels.idx1-ubyte'
    filename_testdata = 'MNIST_data/t10k-images.idx3-ubyte'
    filename_testlabel = 'MNIST_data/t10k-labels.idx1-ubyte'
    train_data = read_image(filename_traindata)/255
    train_label = read_label(filename_trainlabel)
    test_data = read_image(filename_testdata)/255
    test_label = read_label(filename_testlabel)
    # 自定义网络结构与网络参数
    net = np.array([784,200,100,10])   
    learningRate = 2 #学习率
    batch = 100 #batch大小
    epochs = 100 #迭代次数
    dropoutFraction = 0.05 #dropout率
    # 初始化网络
    nn = nn_setup(net,learningRate = learningRate,batch = batch,epochs = epochs)
    
    ##----------------------训练----------------------------
    for epochs in range(nn.epochs):
        time_start = datetime.now()  #记录训练开始时间
        num = int(np.floor(train_data.shape[0]/nn.batch))
        for num_batch in range(num) :
            choose = np.random.randint(1,train_data.shape[0],nn.batch) 
            batch_x = train_data[choose,:]
            batch_y = train_label[choose,:]      
    ##--------------------nn前向传播计算各层输出值---------------
            m = batch_x.shape[0]
            nn.a[0] = np.hstack((np.ones([m,1]),batch_x))
            for i in range(1,nn.size-1):
                nn.a[i] = sigmoid(np.dot(nn.a[i-1],nn.W[i-1].T))
                if nn.dropoutFraction > 0:
                    nn.dropoutMask[i] = np.random.rand(nn.a[i].shape[0],nn.a[i].shape[1])
                    nn.dropoutMask[i][nn.dropoutMask[i] > nn.dropoutFraction] = 1
                    nn.dropoutMask[i][nn.dropoutMask[i] <= nn.dropoutFraction] = 0
                    nn.a[i] = nn.a[i] * nn.dropoutMask[i]
                            
                nn.a[i] = np.hstack((np.ones([m,1]),nn.a[i]))
                
            nn.a[nn.size-1] = sigmoid(np.dot(nn.a[nn.size-2],nn.W[nn.size-2].T))
            nn.e = batch_y - nn.a[nn.size-1] #误差计算
            nn.L = 1/2 * np.sum(nn.e * nn.e)/m
    ##---------------------nn反向传播计算各层梯度----------------
            nn.d[nn.size-1] = - nn.e * (nn.a[nn.size-1] * (1 - nn.a[nn.size-1]))
            for i in range(nn.size-2,0,-1):
                d_act = nn.a[i] * (1 - nn.a[i])
                if i+1 == nn.size-1:
                    nn.d[i] = np.dot(nn.d[i+1],nn.W[i]) * d_act
                else:
                    nn.d[i] = np.dot(nn.d[i+1][:,1:],nn.W[i]) * d_act
                if nn.dropoutFraction > 0:
                    nn.d[i] = nn.d[i] * np.hstack((np.ones([nn.d[i].shape[0],1]),nn.dropoutMask[i]))
                
            for i in range(nn.size-2):
                if i+1 == nn.size-1:
                    nn.dW[i] = np.dot(nn.d[i + 1].T , nn.a[i]) / nn.d[i + 1].shape[0]
                else:
                    nn.dW[i] = np.dot(nn.d[i + 1][:,1:].T , nn.a[i]) / nn.d[i + 1].shape[0]
    ##-------------------nn计算各层梯度更新-------------------
            for i in range(nn.size-2):
                dW = nn.dW[i]
                dW = nn.learningRate * dW
                nn.W[i] = nn.W[i] - dW
            # 相关结果输出
            if num_batch % 100 == 0:
                print('epochs = ', epochs,' / ', nn.epochs,
                        '; batch = ',num_batch,' / ',num,
                        '; error_batch = ', nn.L)       
        time_end = datetime.now()
        print('time using for this epoch = ', (time_end.minute -time_start.minute)*60 + 
              (time_end.second-time_start.second) + 
            (time_end.microsecond - time_start.microsecond)/1000000, 's')
##--------------模型保存至model文件下的model------------------              
    if not os.path.exists('model'):
        os.mkdir('model')
    print('saving model ... ')
    output = open('model/model.pkl', 'wb')
    pickle.dump(nn, output)
    output.close()
            