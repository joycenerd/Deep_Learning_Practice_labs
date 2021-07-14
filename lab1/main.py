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

class Model():
    def __init__(self,input_size,layer1,layer2,lr):
        self.input_size=input_size
        self.layer1=layer1
        self.layer2=layer2
        self.lr=lr
    
        self.w1=np.random.normal(0,1,(self.input_size,self.layer1))
        self.w1=np.random.normal(0,1,(self.layer1,self.layer2))





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


