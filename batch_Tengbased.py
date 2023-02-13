import argparse
import os
from time import time as t
from bindsnet import encoding as ed
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
from bindsnet import ROOT_DIR
from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights


traindata_size = 800
testdata_size = 200


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--n_test", type=int, default=testdata_size)
parser.add_argument("--n_train", type=int, default=traindata_size)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--n_updates", type=int, default=10)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.5)
parser.add_argument("--time", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=True)

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
gpu = args.gpu

update_steps = int(n_train / batch_size / n_updates)
update_interval = update_steps * batch_size

device = "cpu"
# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    # torch.cuda.manual_seed_all(seed)
    pass
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  # gpu * 1 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

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
)

# Directs network to GPU
if gpu:
    network.to("cuda")
test_target_mem=[]
test_predict_mem=[]
############################################################################
from scipy.io import  loadmat
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from spikingjelly.activation_based import encoding
from snntorch import spikegen
file='C:\\Users\\Admin\\Desktop\\dataverse_files\\_50words.mat'
data = loadmat(file, mat_dtype=True)
min_max_scaler = preprocessing.MinMaxScaler()
Z_scaler = preprocessing.StandardScaler()
pe = encoding.PoissonEncoder()
train_data = np.empty(shape=(10,80,15,200))
test_data = np.empty(shape=(10,20,15,200))
train_label = np.empty(shape=(10,80))
test_label = np.empty(shape=(10,20))
train_data_co = np.empty(shape=(traindata_size,time,15,200))
test_data_co = np.empty(shape=(testdata_size,time,15,200))
index = [6,7,8,9,10,11,12,13,14,15]

for i in range(10,20):
# for i in [6,7,8,15,17,23,24,28,36,45]:
    num=str(i)
    num='X'+num
    word = data[num]
    i=i-10
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
# train_data = np.round(train_data)
# test_data = np.round(test_data)
#归一化
for i in range(traindata_size):
    train_data[i] = (train_data[i]-np.min(train_data[i]))/(np.max(train_data[i])-np.min(train_data[i]))
for i in range(testdata_size):
    test_data[i] = (test_data[i]-np.min(test_data[i]))/(np.max(test_data[i])-np.min(test_data[i]))
# 泊松
for i in range(traindata_size):
    train_data_co[i] = spikegen.rate(torch.from_numpy(train_data[i]), num_steps= time, gain = 1).float()
    # train_data_co[i] = torch.repeat_interleave(train_data[i], num_steps, dim = 0).reshape(20,15,200)
for i in range(testdata_size):
    test_data_co[i] = spikegen.rate(torch.from_numpy(test_data[i]), num_steps= time, gain = 1).float()
    # test_data_co[i] = torch.repeat_interleave(test_data[i], num_steps, dim = 0).reshape(20,15,200)
# 泊松
# for i in range(traindata_size):
#     print(i)
#     train_data_co[i] = ed.poisson(torch.from_numpy(train_data[i]*128*5).float(), time = time, dt = dt, device = "cpu", approx=False).numpy()

# for i in range(testdata_size):
#     print(i)
#     test_data_co[i] = ed.poisson(torch.from_numpy(test_data[i]*128*5).float(), time = time, dt = dt, device = "cpu", approx=False).numpy()
#     # test_data_co[i] = ed.repeat(torch.from_numpy(test_data[i]*3).float(), time = time, dt = dt).numpy()

print(train_data_co.shape) #[traindata_size,time,15,200]
print(test_data_co.shape)
# plt.figure()
# plt.tight_layout()
# for i in range(int(traindata_size/80)):
#     for j in range(time):
#         train_data_co[i*80][0] += train_data_co[i*80][j]
#     plt.subplot(10,5,i+1)
#     plt.imshow(train_data_co[i*80][0])
#     plt.pause(0.1)
#     print(i)
##############################################################

# Neuron assignments and spike proportions.
n_classes = 10
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


    if epoch % progress_interval == 0:
        print("\nProgress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

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

        # Optionally plot various simulation information.
        if plot:
            image = batch["image"][:, 0].view(28, 28)
            inpt = inputs["X"][:, 0].view(time, 784).sum(0).view(28, 28)
            lable = batch["label"][0]
            input_exc_weights = network.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {
                layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
            }
            voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=lable, axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(
                accuracy, x_scale=update_steps * batch_size, ax=perf_ax
            )
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )

            plt.pause(1e-8)

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


print(len(test_target_mem),len(test_predict_mem))
sns.set()

C2 = confusion_matrix(test_target_mem,test_predict_mem,labels=range(0,10))
M = np.sum(C2,axis=1).reshape(-1,1)
print(M)
C2=C2/M
sns.heatmap(C2,annot= False,vmin = 0, vmax = 1, fmt='.2f',cmap='Blues', cbar= True) #heatmap
plt.title('Confusion Matrix') #title
plt.xlabel('Prediction') #x axis
plt.ylabel('Label') #y  axis
plt.savefig('./full_connect_rate.png')
