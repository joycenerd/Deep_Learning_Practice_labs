import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    plt.rcParams["font.family"] = "serif"
    X = np.linspace(1, 10, 10, endpoint=True)

    resnet50_dict = {
        "Train(with pretraining)": "./results/acc/resent50_pretrained_train_acc.csv",
        'Test(with pretraining)': './results/acc/resent50_pretrained_eval_acc.csv',
        "Train(w/o pretraining)": "./results/acc/resnet50_train_acc.csv",
        'Test(w/o pretraining)': './results/acc/resnet50_eval_acc.csv'
    }

    plt.figure()

    for label in resnet50_dict:
        filename = resnet50_dict[label]
        data = pd.read_csv(filename)
        acc = data['Value']
        acc *= 100
        plt.plot(X, acc, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.title("Result comparison(ResNet50)")
    plt.savefig("./results/resnet50_acc.jpg")

    resnet18_dict = {
        "Train(with pretraining)": "./results/acc/resent18_pretrained_train_acc.csv",
        'Test(with pretraining)': './results/acc/resent18_pretrained_eval_acc.csv',
        "Train(w/o pretraining)": "./results/acc/resnet18_train_acc.csv",
        'Test(w/o pretraining)': './results/acc/resnet18_eval_acc.csv'
    }

    plt.figure()

    for label in resnet18_dict:
        filename = resnet18_dict[label]
        data = pd.read_csv(filename)
        acc = data['Value']
        acc *= 100
        plt.plot(X, acc, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.title("Result comparison(ResNet18)")
    plt.savefig("./results/resnet18_acc.jpg")
