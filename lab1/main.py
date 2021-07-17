import numpy as np
import argparse
import os 


parser=argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="linear", help="which kind of data to process: [linear, xor]")
args=parser.parse_args()


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


def sigmoid(x):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
    """
    return 1 / (1 + np.exp(-x))


def der_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that output from sigmoid function.
    """
    return y * (1 - y)

class Model:
    def __init__(self, hidden_size, epochs, lr):
        """ Feedforward network with 2 hidden layers
        Args:
            hidden_size: two hidden layers neurons (layer1,layer2)
            epochs: num of training epochs
        """
        self.epochs = epochs

        # Model parameters initialization
        input_size = 2
        output_size = 1
        self.lr = lr
        self.momentum = 0.9
        (layer1,layer2)=hidden_size
        layer1=10
        layer2=10

        self.w1 = np.random.randn(input_size, layer1)
        self.w2 = np.random.randn(layer1, layer2)
        self.w3 = np.random.randn(layer2, output_size)
        self.b1 = np.zeros((1, layer1))
        self.b2 = np.zeros((1, layer2))
        self.b3 = np.zeros((1, output_size))
        
        self.v_w1 = np.zeros((input_size, layer1) )
        self.v_w2 = np.zeros((layer1, layer2))
        self.v_w3 = np.zeros((layer2, output_size))
        self.v_b1 = np.zeros((1, layer1))
        self.v_b2 = np.zeros((1, layer2))
        self.v_b3 = np.zeros((1, output_size))

    @staticmethod
    def plot_result(data, gt_y, pred_y):
        """ Data visualization with ground truth and predicted data comparison. There are two plots
        for them and each of them use different colors to differentiate the data with different labels.
        Args:
            data:   the input data
            gt_y:   ground truth to the data
            pred_y: predicted results to the data
        """
        assert data.shape[0] == gt_y.shape[0]
        assert data.shape[0] == pred_y.shape[0]

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for idx in range(data.shape[0]):
            if gt_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Prediction', fontsize=18)

        for idx in range(data.shape[0]):
            if pred_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.show()

    def forward(self, inputs):
        # Forward pass
        self.input = inputs
        self.a1    = sigmoid(np.dot(self.input, self.w1)+self.b1)
        self.a2    = sigmoid(np.dot(self.a1, self.w2)+self.b2)
        output     = sigmoid(np.dot(self.a2, self.w3)+self.b3)

        return output

    def backward(self):
        # backward pass
        dout    = self.error
        dout    = np.multiply(dout, der_sigmoid(self.output))
        grad_w3 = np.dot(self.a2.T, dout)
        grad_b3 = np.sum(dout, axis=0)

        dout    = np.dot(dout, self.w3.T)
        dout    = np.multiply(dout, der_sigmoid(self.a2))
        grad_w2 = np.dot(self.a1.T, dout)
        grad_b2 = np.sum(dout, axis=0)

        dout    = np.dot(dout, self.w2.T)
        dout    = np.multiply(dout, der_sigmoid(self.a1))
        grad_w1 = np.dot(self.input.T, dout)
        grad_b1 = np.sum(dout, axis=0)
        
        
        self.v_w1 = self.momentum * self.v_w1 + self.lr * grad_w1 
        self.v_w2 = self.momentum * self.v_w2 + self.lr * grad_w2
        self.v_w3 = self.momentum * self.v_w3 + self.lr * grad_w3
                                                           
        self.v_b1 = self.momentum * self.v_b1 + self.lr * grad_b1
        self.v_b2 = self.momentum * self.v_b2 + self.lr * grad_b2
        self.v_b3 = self.momentum * self.v_b3 + self.lr * grad_b3

        self.w1 -= self.v_w1
        self.w2 -= self.v_w2
        self.w3 -= self.v_w3
                            
        self.b1 -= self.v_b1
        self.b2 -= self.v_b2
        self.b3 -= self.v_b3

        return

    def train(self, inputs, labels):
        """ model training
        Args:
            inputs: the training (and testing) data used in the model.
            labels: the ground truth of correspond to input data.
        """
        # make sure that the amount of data and label is match
        assert inputs.shape[0] == labels.shape[0]

        n = inputs.shape[0]
        self.pre_error = 1000
        error = 0

        for epochs in range(self.epochs):

            for idx in range(n):
                self.output = self.forward(inputs[idx:idx+1, :])
                self.error = self.output - labels[idx:idx+1, :]
                self.backward()

            train_loss,train_acc=self.test(inputs,labels)
            train_loss=train_loss.squeeze().squeeze()
            print(f"Epoch {epochs+1}/{self.epochs} train_loss: {train_loss:.4f} train_acc: {train_acc:.4f}")
                   
            if train_loss > self.pre_error:
                self.lr *= 0.9
                pass
            self.pre_error = error

        print('Training finished')
        test_loss,test_acc=self.test(inputs, labels,True)
        test_loss=test_loss.squeeze().squeeze()
        print(f"test_loss: {test_loss:.4f}\ttest_acc: {test_acc:.4f}")


    def test(self, inputs, labels,print_res=False):
        """ testing
        Args:
            inputs: the testing data. One or several data samples are both okay.
                The shape is expected to be [BatchSize, 2].
            labels: the ground truth correspond to the inputs.
        """
        n = inputs.shape[0]
        error = 0
        acc = 0
        all_result=[]

        for idx in range(n):
            result = self.forward(inputs[idx:idx+1, :])
            all_result.append(result)
            error += abs(result - labels[idx:idx+1, :])
            acc += (result[0][0] >= 0.5) == labels[idx:idx+1, :][0][0]

        error /= n
        acc /= n
        if print_res:
            for res in all_result:
                print(res)

        return error, acc


if __name__=="__main__":
    task=args.task

    # generate data
    if task=="linear":
        if not os.path.exists("./data/linear_data.csv"):
            data,label=generate_linear(n=100)
            data=np.append(data,label,axis=1)
            np.savetxt("./data/linear_data.csv",data,delimiter=",")
        else:
            all_data = np.genfromtxt("./data/linear_data.csv", delimiter=",")
            data,label = all_data[:,:2], all_data[:,2]
            label=label.astype(int).reshape(-1,1)
    if task == "xor":
        if not os.path.exists("./data/xor_data.csv"):
            data,label=generate_XOR_easy()
            data=np.append(data,label,axis=1)
            np.savetxt("./data/xor_data.csv",data,delimiter=",")
        else:
            all_data = np.genfromtxt("./data/xor_data.csv", delimiter=",")
            data,label = all_data[:,:2], all_data[:,2]
            label=label.astype(int).reshape(-1,1)

    model=Model((5,10),1000,0.1)
    model.train(data,label)



