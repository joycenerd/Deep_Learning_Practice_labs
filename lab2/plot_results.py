import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__=="__main__":
    # plot EEG
    EEGNet_dict={
        'elu':'elu_5e-3_amsgrad',
        'relu':'relu_1e-3',
        'leaky_relu':'leaky_relu_1e-2_init_amsgrad'
    }

    plt.figure()
    plt.rcParams["font.family"]="serif"
    X=np.linspace(1,500,500,endpoint=True)
    
    for act in EEGNet_dict:
        name=EEGNet_dict[act]
        filename="./results/EEGNet/EEGNet_"+name+'-train_acc.csv'
        data=pd.read_csv(filename)
        acc=data['Value']
        acc*=100
        plt.plot(X,acc,label=act+'_train')

        filename="./results/EEGNet/EEGNet_"+name+'-eval_acc.csv'
        data=pd.read_csv(filename)
        acc=data['Value']
        acc*=100
        plt.plot(X,acc,label=act+'_test')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.title("Activation function comparison(EEGNet)")
    plt.savefig("./results/EEGNet/EEGNet_acc.jpg") 


    # plot DeepConvNet
    DeepConvNet_dict={
        'elu':'elu_1e-3_amsgrad',
        'relu':'relu_1e-2',
        'leaky_relu':'leaky_relu_1e-2_init_amsgrad'
    }

    plt.figure()
    plt.rcParams["font.family"]="serif"
    X=np.linspace(1,500,500,endpoint=True)

    
    for act in DeepConvNet_dict:
        name=DeepConvNet_dict[act]
        filename="./results/DeepConvNet/DeepConvNet_"+name+'-train_acc.csv'
        data=pd.read_csv(filename)
        acc=data['Value']
        acc*=100
        plt.plot(X,acc,label=act+'_train')

        filename="./results/DeepConvNet/DeepConvNet_"+name+'-eval_acc.csv'
        data=pd.read_csv(filename)
        acc=data['Value']
        acc*=100
        plt.plot(X,acc,label=act+'_test')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.title("Activation function comparison(DeepConvNet)")
    plt.savefig("./results/DeepConvNet/DeepConvNet_acc.jpg")         

