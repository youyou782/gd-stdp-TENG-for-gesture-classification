import argparse
import os
from os import path
from scipy.io import loadmat
from snntorch import spikegen
from time import time as t
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor

def load_data(train_data_number, test_data_number, class_number):

    ###load training data
    # allocate RAM
    train_data = np.empty(shape=(class_number,80,15,200))
    test_data = np.empty(shape=(class_number,20,15,200))
    train_label = np.empty(shape=(class_number,80))
    test_label = np.empty(shape=(class_number,20))

    train_data_co = np.empty(shape=(train_data_number,time,15,200))
    test_data_co = np.empty(shape=(test_data_number,time,15,200))
    train_label_co= []
    test_label_co = []

    file=path.dirname(path.dirname(__file__))+'/_50words.mat'
    data = loadmat(file, mat_dtype=True)

    #choose classes here
    index = [5,20,2,23,15,16,6,48,8,9]
    for i in range(0,class_number):

        word = data['X'+ str(index[i])]

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
        train_data_co[i] = spikegen.rate(train_data[i], num_steps= time, gain = 0.1)
    for i in range(test_data_number):
        test_data_co[i] = spikegen.rate(test_data[i], num_steps= time, gain = 0.1)
    print("data load finished")
    return train_data_co, train_label_co, test_data_co, test_label_co


traindata_size = 800
testdata_size = 200
n_classes = 10

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--n_epochs", type=int, default=3)
parser.add_argument("--n_test", type=int, default=testdata_size)
parser.add_argument("--n_train", type=int, default=traindata_size)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--n_updates", type=int, default=3) 
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.5)
parser.add_argument("--time", type=int, default=200)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.set_defaults(plot=False)

args = parser.parse_args()
seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
n_updates = args.n_updates
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
train = args.train
plot = args.plot

update_steps = int(n_train / batch_size / n_updates)
update_interval = update_steps * batch_size

device = "cpu"
# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    gpu = True
else:
    gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use 
if n_workers == -1:
    n_workers = 0  # gpu * 1 * torch.cuda.device_count()

# Build network.
network = DiehlAndCook2015(
    n_inpt=3000,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=300,
    nu=(4.5e-4, 4.5e-2),
    theta_plus=theta_plus,
    inpt_shape=(15, 200),
).to(device)

# Directs network to GPU
if gpu:
    network.to("cuda")

test_target_mem=[]
test_predict_mem=[]

train_data_co, train_label_co, test_data_co, test_label_co = load_data(train_data_number = traindata_size, test_data_number = testdata_size, class_number = n_classes)

# plt.figure()
# plt.tight_layout()
# for i in range(int(traindata_size/80)):
#     for j in range(time):
#         train_data_co[i*80][0] += train_data_co[i*80][j]
#     plt.subplot(10,5,i+1)
#     plt.imshow(train_data_co[i*80][0])
#     plt.pause(0.1)
#     print(i)

# Neuron assignments and spike proportions.
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Ai"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin training...")
start = t()

test_data = test_data_co.reshape(int(testdata_size/batch_size), batch_size, time, 15, 200)
test_label = test_label_co.reshape(int(testdata_size/batch_size),batch_size)
for epoch in range(n_epochs):
    labels = []
    # reload data and label
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

    pbar_training = tqdm(total=n_train)
    for step in range(int(traindata_size/batch_size)):
        if step * batch_size > n_train:
            break
        batch = {"encoded_image":torch.from_numpy(train_data[step].transpose(1,0,2,3)),"label":train_label[step]}
        # Assign labels to excitatory neurons.
        if step % update_steps == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        # Get next input sample.
        inputs = {"X": batch["encoded_image"]}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Remember labels.
        labels.extend(batch["label"].tolist())

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Add to spikes recording.
        s = spikes["Ae"].get("s").permute((1, 0, 2))
        spike_record[
            (step * batch_size)
            % update_interval : (step * batch_size % update_interval)
            + s.size(0)
        ] = s

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")
        network.reset_state_variables()  # Reset state variables.
        pbar_training.update(batch_size)
    pbar_training.close()

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("\nTraining complete.\n")


# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Train the network.
print("\nBegin testing...\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
pbar.set_description_str("Test progress: ")

for step in range(int(testdata_size/batch_size)):
    if step * batch_size > n_test:
        break
    # Get next input sample.
    batch={"encoded_image":torch.from_numpy(test_data[step].transpose(1,0,2,3)),"label":test_label[step]}
    inputs = {"X": batch["encoded_image"]}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # Add to spikes recording.
    spike_record = spikes["Ae"].get("s").permute((1, 0, 2))

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    test_target_mem.extend(label_tensor.tolist())
    test_predict_mem.extend(all_activity_pred.tolist())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.update(batch_size)
pbar.close()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("\nTesting complete.\n")

sns.set()

C2 = confusion_matrix(test_target_mem,test_predict_mem,labels=range(0,10))
M = np.sum(C2,axis=1).reshape(-1,1)
print(M)
C2=C2/M
sns.heatmap(C2,annot= False,vmin = 0, vmax = 1, fmt='.2f',cmap='Blues', cbar= True) #heatmap
plt.title('Confusion Matrix') #title
plt.xlabel('Prediction') #x axis
plt.ylabel('Label') #y  axis
plt.savefig('./stdp-SNN.png')
