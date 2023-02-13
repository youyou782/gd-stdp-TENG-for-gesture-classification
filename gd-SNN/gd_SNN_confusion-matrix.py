from os import path 

import datetime
import snntorch as snn
import numpy as np

from snntorch import spikegen
from snntorch import surrogate
from snntorch import functional as SF

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.io import  loadmat

import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_data(train_data_number, test_data_number):

    ###load training data
    # allocate RAM
    train_data = np.empty(shape=(50,80,15,200))
    test_data = np.empty(shape=(50,20,15,200))
    train_label = np.empty(shape=(50,80))
    test_label = np.empty(shape=(50,20))

    train_data_co = np.empty(shape=(train_data_number,time,15,200))
    test_data_co = np.empty(shape=(test_data_number,time,15,200))
    train_label_co= []
    test_label_co = []

    file=path.dirname(path.dirname(__file__))+'/_50words.mat'
    data = loadmat(file, mat_dtype=True)
    for i in range(0,50):

        word = data['X'+ str(i)]

        word = (np.array(word)).transpose(1,0)                    #(number_of_samples, 3000)
        word = word.reshape(np.size(word,0), 200, 15)             #(number of samples, time_steps, channels)
        word = word.transpose(0,2,1)                              #(number of samples, channels, time_steps)

        a = np.split(word,[80,100])                               #(split data)
        b = np.split(np.full((100),np.array(i)),[80])             #(generate and split labels)

        train_data[i] = a[0]                                      # assign (0~80, 15, 200) for training
        train_label[i] = b[0]

        test_data[i] = a[1]                                       #assign (81~100, 15, 200) for testing
        test_label[i] = b[1]
    train_data = train_data.reshape(train_data_number,15,200)     #reshape for standardlization
    test_data = test_data.reshape(test_data_number,15,200)
    train_label_co = train_label.reshape(train_data_number,1)
    test_label_co = test_label.reshape(test_data_number,1)
    # standardlization: minus average value in each channel respectively and use the absolute value.
    train_data = train_data.transpose(1,0,2)
    test_data = test_data.transpose(1,0,2)
    for i in range(15):
        train_data[i] = train_data[i]-np.mean(train_data[i])
    for i in range(15):
        test_data[i] = test_data[i]-np.mean(test_data[i])
    train_data = np.abs(train_data.transpose(1,0,2))
    test_data = np.abs(test_data.transpose(1,0,2))
    # normalization to [0,1]
    for i in range(train_data_number):
        train_data[i] = (train_data[i]-np.min(train_data[i]))/(np.max(train_data[i])-np.min(train_data[i]))
    for i in range(test_data_number):
        test_data[i] = (test_data[i]-np.min(test_data[i]))/(np.max(test_data[i])-np.min(test_data[i]))

    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)

    #Choose to round the data to the nearest intger or not
    # train_data = np.round(train_data)
    # test_data = np.round(test_data)

    # Apply a Poission Encoder
    for i in range(train_data_number):
        train_data_co[i] = spikegen.rate(train_data[i], num_steps= time, gain = 1)
    for i in range(test_data_number):
        test_data_co[i] = spikegen.rate(test_data[i], num_steps= time, gain = 1)
    print("data load finished")
    return train_data_co, train_label_co, test_data_co, test_label_co


# set parameters(tune parameter below)
batch_size = 50
# surrogate gradient
spike_grad = surrogate.fast_sigmoid(slope=25)
# Network Architecture
num_hidden = 5640
num_output = 50
# Temporal Dynamics  
time = 20
beta = 0.7

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("running on : " + str(device))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize layers
        self.conv1 = nn.Conv1d(in_channels = 15,out_channels =60, kernel_size = (5))
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv1d(in_channels = 60,out_channels = 120, kernel_size = (5))
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(num_hidden, num_output)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        # Record the final layer
        spk3_rec = []
        mem3_rec = []
        
        for step in range(time):
            cur1 = F.max_pool1d(self.conv1(x[step]), 2)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = F.max_pool1d(self.conv2(spk1), 2)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.fc1(spk2.view(batch_size,-1))
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
        

# Load the network onto CUDA if available
net = Net().to(device)

#choose to load the pre-trained model or not
net.load_state_dict(torch.load(path.dirname(__file__)+'/bp_SNN_net.pt'))

torch.set_printoptions(profile="full")

train_data_number = 4000
test_data_number = 1000
#_co represents the data being coded into spikes

train_data_co, train_label_co, test_data_co, test_label_co = load_data(train_data_number = train_data_number, test_data_number = test_data_number)

# begin test 
test_target_mem=[]
test_predict_mem=[]
# prepare test data
test_data = test_data_co.reshape(int(test_data_number/batch_size), batch_size, time, 15, 200)
test_label = test_label_co.reshape(int(test_data_number/batch_size),batch_size)

for epoch in range(1):
    with torch.no_grad():
        net.eval()

        # testset accuracy
        test_accuracy = 0
        for j in range(0, int(1000/batch_size)): 
            data = torch.from_numpy(test_data[j]).float().permute(1,0,2,3).to(device)
            targets = torch.from_numpy(test_label[j]).long().to(device)
            test_spk, test_mem = net(data)
            test_target_mem.extend(targets.tolist())
            
            test_spk=test_spk.transpose(0,1)
            for num in range(batch_size):
                index = torch.argmax(torch.sum(test_spk[num],dim=0))
                test_predict_mem.append(index.item())
        break

plt.figure(dpi=1000,figsize=(6.4,4.8))
sns.set()
C2 = confusion_matrix(test_target_mem,test_predict_mem,labels=range(0,50))
M = np.sum(C2,axis=1).reshape(-1,1)
print(M)
C2=C2/M
sns.heatmap(C2,annot= True,vmin = 0, vmax = 1, fmt='.2f',cmap='Blues', cbar= True, annot_kws={"fontsize":2}) #heatmap
plt.title('Confusion Matrix') #title
plt.xlabel('Prediction') #x axis
plt.ylabel('Label') #y  axis
plt.savefig(path.dirname(__file__) + '/gd_SNN_matrix.png')

