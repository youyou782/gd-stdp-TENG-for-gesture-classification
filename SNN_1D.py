def plot_acc_curve():
    plt.figure(1)
    plt.axis([0, epoch, 0.6 , 1])
    plt.xlabel('Epoch')
    plt.title('Train/Test Accuracy')
    plt.plot(np.linspace(0,epoch,epoch+1),train_accuracy_hist)
    plt.plot(np.linspace(0,epoch,epoch+1),test_accuracy_hist)
    plt.legend(['Train Accuarcy','Test Accuracy'])
    plt.savefig('pic.png')
    plt.show()

def save_var(train_accuracy_hist, test_accuracy_hist, loss_hist, test_loss_hist):
    # save the acc 
    torch.save(net.state_dict(), "./net.pt")
    train_accuracy_hist=np.array(train_accuracy_hist)
    np.save('train_accuracy_hist.npy',train_accuracy_hist)
    test_accuracy_hist=np.array(test_accuracy_hist)
    np.save('test_accuracy_hist.npy',test_accuracy_hist)

    # save the loss
    loss_hist=np.array(loss_hist)
    np.save('loss_hist.npy',loss_hist)
    test_loss_hist=np.array(test_loss_hist)
    np.save('test_loss_hist.npy',test_loss_hist)

def cal_accuracy(y,test_target):
    #y is the output size[(T, Batch_size, Output)]
    #test_target is the corresponding index size[(Batch_size)]
    a = 0.0
    y=y.transpose(0,1)
    for num in range(batch_size):
        index = torch.argmax(torch.sum(y[num],dim=0))
        if index == test_target[num]:
            a = a + 1
    return a
import datetime
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import  loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import torch

# dataloader arguments
batch_size = 10

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
# Define a transform

spike_grad = surrogate.fast_sigmoid(slope=25)
# Network Architecture
num_hidden = 2520
num_output = 50
num_dense = 300
# Temporal Dynamics
num_steps = 20
time = num_steps
dt = 1.0
beta = 0.7

num_epochs = 100
loss_hist = []
test_loss_hist = []
iteration = 0

train_accuracy_hist=[]
test_accuracy_hist=[]

test_num_right = 0  
train_num_right = 0
# torch.backends.cudnn.enabled = False
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        # self.bn = nn.BatchNorm1d(num_features = 15)
        self.conv1 = nn.Conv1d(in_channels = 15,out_channels =60, kernel_size = (5))
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv1d(in_channels = 60,out_channels = 120, kernel_size = (5))
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # self.conv3 = nn.Conv1d(in_channels = 60,out_channels = 120, kernel_size = (5))
        # self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # self.fc1 = nn.Linear(num_hidden, num_dense)
        # self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(5640, 50)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # time_1=datetime.datetime.now()
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        # mem3 = self.lif3.init_leaky()
        # mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        # Record the final layer
        spk5_rec = []
        mem5_rec = []
        
        for step in range(num_steps):
            cur1 = F.max_pool1d(self.conv1(x[step]), 2)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = F.max_pool1d(self.conv2(spk1), 2)
            spk2, mem2 = self.lif2(cur2, mem2)

            # cur3 = F.max_pool1d(self.conv3(spk2), 2)
            # spk3, mem3 = self.lif3(cur3, mem3)

            # cur4 = self.fc1(spk3.view(batch_size, -1))
            # spk4, mem4 = self.lif4(cur4, mem4)
            
            cur5 = self.fc2(spk2.view(batch_size,-1))
            spk5, mem5 = self.lif5(cur5, mem5)

            spk5_rec.append(spk5)
            mem5_rec.append(mem5)
        # print(datetime.datetime.now()-time_1)
        return torch.stack(spk5_rec, dim=0), torch.stack(mem5_rec, dim=0)
        

# Load the network onto CUDA if available
net = Net().to(device)
#net.load_state_dict(torch.load("/net.pt"))

loss = SF.mse_count_loss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-5, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',verbose=True, patience = 20, factor = 0.2)
torch.set_printoptions(profile="full")
####################################################################################
traindata_size = 4000
testdata_size = 1000
from bindsnet import encoding as ed
file='C:\\Users\\Admin\\Desktop\\dataverse_files\\_50words.mat'
data = loadmat(file, mat_dtype=True)
min_max_scaler = preprocessing.MinMaxScaler()
Z_scaler = preprocessing.StandardScaler()
train_data = np.empty(shape=(50,80,15,200))
test_data = np.empty(shape=(50,20,15,200))
train_label = np.empty(shape=(50,80))
test_label = np.empty(shape=(50,20))
train_data_co = np.empty(shape=(traindata_size,time,15,200))
test_data_co = np.empty(shape=(testdata_size,time,15,200))

for i in range(0,50):
    num=str(i)
    num='X'+num
    word = data[num]

    worddata = np.array(word) #(3000, Samples,)
    worddata = worddata.transpose(1,0) #(Samples, 3000)
    a = np.size(worddata,0)
    worddata = worddata.reshape(a, 200, 15)
    worddata = worddata.transpose(0,2,1) #(115,15,200)

    a = np.split(worddata,[80,100])
    b = np.split(np.full((100),np.array(i)),[80])

    train_data[i] = a[0] # (80, 15, 200)
    train_label[i] = b[0]

    test_data[i] = a[1] #(20, 15, 200)
    test_label[i] = b[1]

train_data = train_data.reshape(traindata_size,15,200)
test_data = test_data.reshape(testdata_size,15,200)
train_label_co = train_label.reshape(traindata_size,1)
test_label_co = test_label.reshape(testdata_size,1)
# 每个通道减均值
train_data = train_data.transpose(1,0,2)
test_data = test_data.transpose(1,0,2)
for i in range(15):
    train_data[i] = train_data[i]-np.mean(train_data[i])
for i in range(15):
    test_data[i] = test_data[i]-np.mean(test_data[i])
train_data = np.abs(train_data.transpose(1,0,2))
test_data = np.abs(test_data.transpose(1,0,2))
# 归一化
for i in range(traindata_size):
    train_data[i] = (train_data[i]-np.min(train_data[i]))/(np.max(train_data[i])-np.min(train_data[i]))
for i in range(testdata_size):
    test_data[i] = (test_data[i]-np.min(test_data[i]))/(np.max(test_data[i])-np.min(test_data[i]))
train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
# train_data = torch.from_numpy(np.round(train_data))
# test_data = torch.from_numpy(np.round(test_data))

# 泊松
for i in range(traindata_size):
    print(i)
    train_data_co[i] = spikegen.rate(train_data[i], num_steps= num_steps, gain = 1)
    # train_data_co[i] = torch.repeat_interleave(train_data[i], num_steps, dim = 0).reshape(20,15,200)
for i in range(testdata_size):
    print(i)
    test_data_co[i] = spikegen.rate(test_data[i], num_steps= num_steps, gain = 1)
    # test_data_co[i] = torch.repeat_interleave(test_data[i], num_steps, dim = 0).reshape(20,15,200)

print(train_data_co.shape) #[traindata_size,time,15,200]
print(test_data_co.shape)
plt.figure()
plt.tight_layout()
for i in range(int(traindata_size/80)):
    for j in range(time):
        train_data_co[i*80][0] += train_data_co[i*80][j]
    plt.subplot(10,5,i+1)
    plt.imshow(train_data_co[i*80][0])
    plt.pause(0.1)
    print(i)
print("data load finished")
####################################################################################

test_data = test_data_co.reshape(int(testdata_size/batch_size), batch_size, time, 15, 200)
test_label = test_label_co.reshape(int(testdata_size/batch_size),batch_size)
for epoch in range(num_epochs):

    train_data = train_data_co
    train_label = train_label_co
    # shuffle in the same sequence
    state = np.random.get_state()
    np.random.shuffle(train_data)
    np.random.set_state(state)
    np.random.shuffle(train_label)
    # turn data into batch
    train_data = train_data.reshape(int(traindata_size/batch_size), batch_size, time, 15, 200)
    train_label = train_label.reshape(int(traindata_size/batch_size), batch_size)

    print(epoch)

    for j in range(0,int(4000/batch_size)):
        net.train()
        #STFT process is in the forward function
        data = torch.from_numpy(train_data[j]).float()
        targets = torch.from_numpy(train_label[j]).long()
        data = data.permute(1,0,2,3).to(device)
        targets = targets.to(device)
        spk_rec, mem_rec = net(data)

        
        # initialize the loss & sum over time
        loss_val = loss(spk_rec, targets)
        print(loss_val)
        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        loss_hist.append(loss_val.item())

    with torch.no_grad():
        net.eval()
        train_accuracy = 0
        for j in range(0,int(4000/batch_size)):
            #STFT process is in the forward function
            data = torch.from_numpy(train_data[j]).float()
            targets = torch.from_numpy(train_label[j]).long()
            data = data.permute(1,0,2,3).to(device)
            targets = targets.to(device)
            spk_rec, mem_rec = net(data)
            
            train_accuracy += cal_accuracy(spk_rec,targets)
        print("train_accracy: ", train_accuracy/(4000))
        train_accuracy_hist.append(train_accuracy/(4000))

        # testset accuracy
        test_accuracy = 0
        for j in range(0, int(1000/batch_size)): 
            test_datas = torch.from_numpy(test_data[j]).float()
            test_targets = torch.from_numpy(test_label[j]).long()
            test_datas = test_datas.permute(1,0,2,3).to(device)
            test_targets = test_targets.to(device)
            test_spk, test_mem = net(test_datas)
            test_accuracy += cal_accuracy(test_spk,test_targets)
        print("test_accracy: ", test_accuracy/(1000))
        test_accuracy_hist.append(test_accuracy/(1000))



    scheduler.step(train_accuracy)
    f = open('./record.txt','a')
    text = ("{0:<5},{1:<30},{2:<10.3f},{3:<7}".format(str(epoch),str(datetime.datetime.now()), test_accuracy/1000,str(optimizer.state_dict()['param_groups'][0]['lr']))) + '\n'
    f.write( text )
    f.close()
torch.save(net.state_dict(), "./net.pt") 

#save varaiable
#save_var(train_accuracy_hist, test_accuracy_hist, loss_hist, test_loss_hist)
#plot
#plot_acc_curve()

