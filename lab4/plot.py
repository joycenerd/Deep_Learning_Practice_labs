import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.family"]="serif"
plt.figure()

res_list=['BLEU4-score','CrossEntropy','Gaussian-Score','KLD_weight','KLD','Teacher ratio']

# Plot KLD
filename='results/cyclic/run-train_KLD-tag-train.csv'
data=pd.read_csv(filename)
x=data['Step']
val=data['Value']
plt.plot(x,val,label='KLD')

# Plot CrossEntropy
filename='results/cyclic/run-train_CrossEntropy-tag-train.csv'
data=pd.read_csv(filename)
x=data['Step']
val=data['Value']
plt.plot(x,val,label='CrossEntropy')

plt.xlabel('500 iteration(s)')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig(f'results/cyclic/cyclic_loss.jpg')

plt.figure()

# Plot bleu
filename='results/cyclic/run-train_BLEU4-score-tag-train.csv'
data=pd.read_csv(filename)
x=data['Step']
val=data['Value']
plt.plot(x,val,'o',label='BLEU4-score',markersize=1)

# Plot KLD weight
filename='results/cyclic/run-train_KLD_weight-tag-train.csv'
data=pd.read_csv(filename)
x=data['Step']
val=data['Value']
plt.plot(x,val,'--',label='KLD_weight')

# Plot tf
filename='results/cyclic/run-train_tf-tag-train.csv'
data=pd.read_csv(filename)
x=data['Step']
val=data['Value']
plt.plot(x,val,'--',label='Teacher ratio')

# Plot Gaussian score
filename='results/cyclic/run-train_Gaussian-Score-tag-train.csv'
data=pd.read_csv(filename)
x=data['Step']
val=data['Value']
plt.plot(x,val,'o',label='Gaussian-Score',markersize=1)

plt.xlabel("500 iteration(s)")
plt.ylabel('score/weight')
plt.title('Training Ratio Curve')
plt.legend()
plt.savefig('results/cyclic/cyclic_ratio.jpg')