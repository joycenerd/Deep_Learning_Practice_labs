import pandas as pd
import numpy as np
import os


def generate_linear(n=100):
    pts=np.random.uniform(0,1,(n,2))
    inputs=[]
    labels=[]
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance=(pt[0]-pt[1])/1.414
        if pt[0]>pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs),np.array(labels).reshape(n,1)


def generate_XOR_easy():
    inputs=[]
    labels=[]
    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)
        if 0.1*i==0.5:
            continue
        inputs.append([0.1*i,1-0.1*i])
        labels.append(1)
    return np.array(inputs),np.array(labels).reshape(21,1)


def softwax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)


def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples,num_classes)
    y is labels (num_examples,1)
    """
    m=y.shape[0]
    p=softwax(X)

    # use multidimensional array indexing to extract softmax probability of the correct label for each sample
    log_likelihood=-np.log(p[range(m),y])
    loss=np.sum(log_likelihood)/m
    return loss


def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples,num_classes)
    y is labels (num_examples,1)
    """
    m = y.shape[0]
    grad = softwax(X)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad


class Model():
    def __init__(self,input_size,layer1,layer2,lr):
        self.input_size=input_size
        self.layer1=layer1
        self.layer2=layer2
        self.lr=lr
    
        self.w1=np.random.normal(0,1,(self.input_size,self.layer1))
        self.w2=np.random.normal(0,1,(self.layer1,self.layer2))
        self.w3=np.random.normal(0,1,(self.layer2,2))
    
    def forward(self,x):
        self.a1=x@self.w1
        self.z1=sigmoid(self.a1)
        self.a2=self.z1@self.w2
        self.z2=sigmoid(self.a2)
        self.a3=self.z2@self.w3
        out = sigmoid(self.a3)
        return out

    def backward(self,X,y):
        grad_y_pred = delta_cross_entropy(X,y)




        
    
    def sigmoid(x):
        return 1.0/(1.0+np.exp(-x))


if __name__=="__main__":
    # generate data
    if not os.path.exists("./data/linear_data.csv"):
        data,label=generate_linear(n=100)
        data=np.append(data,label,axis=1)
        np.savetxt("./data/linear_data.csv",data,delimiter=",")
    if not os.path.exists("./data/xor_data.csv"):
        data,label=generate_XOR_easy()
        data=np.append(data,label,axis=1)
        np.savetxt("./data/xor_data.csv",data,delimiter=",")


