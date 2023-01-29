# numpy: the library that deals with linear algebra calculations
import numpy as np
# (py)torch: our machine learning library!
import torch
import torch.nn.functional as F
from torch import nn, optim, utils
from torchvision import datasets, transforms
# matplotlib: the library for making plots & displaying images
import matplotlib.pyplot as plt
# tqdm: the library for displaying progress bars
from tqdm.notebook import tqdm

# use GPU is GPU (CUDA) is available, use CPU otherwise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# pre-process images
transform = transforms.Compose([transforms.ToTensor()])  # convert image to pytorch tensor
# download images and create PyTorch Dataset & DataLoader objects
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)

# these are the CIFAR-10 classes, 0 for plane, 1 for car, 2 for bird, etc.
classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# data augmentation to prevent overfitting
data_augmentation = nn.Sequential(transforms.RandomHorizontalFlip(),
                                  transforms.RandomCrop((32, 32), padding=(4, 4)))
data_augmentation = torch.jit.script(data_augmentation)  # scriptify to make augmentation run faster


class CNN(nn.Module):

    def __init__(self):
        """Initialize the CNN network and define its layers"""
        super().__init__()

        # there are many different ways of defining a PyTorch model, nn.Sequential is just the most
        # convenient one as everything is in one callable 'thing' and executes in sequence
        self.network = nn.Sequential(
            # layer 1: b x 3 x 32 x 32 -> b x 32 x 16 x 16 (format: channel x width x height)
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # layer 2: b x 32 x 16 x 16 -> b x 64 x 8 x 8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # layer 3: b x 64 x 8 x 8 -> b x 64 x 4 x 4
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # flatten and output fully-connected layer
            nn.Flatten(),  # b x 64 x 4 x 4 -> b x (64 * 4 * 4)
            nn.Linear(64 * 4 * 4, 10))  # b x (64 * 4 * 4) -> b x 10
        # ... however, you can do something like:
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        # self.relu1 = nn.ReLU()
        # sekf.batch_norm1 = nn.BatchNorm2d(32)
        # ...
        # and then in the forward function sequentially pass x through these class variables

    def forward(self, x):
        """Pass input x through the network to obtain an output"""
        output = self.network(x)
        return output

    def size(self):
        """Count the number of parameters (i.e., size of weights) in the network"""
        parameters = self.parameters()
        size = 0
        for parameter in parameters:
            size += np.prod(parameter.shape)
        return size

network = CNN()
network.to(device)  # if we are using GPU, put the network weights on the GPU

print(f"Network size: {network.size()}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.NAdam(network.parameters(), lr=0.0003, weight_decay=1e-5)

@torch.no_grad()  # we don't want to compute gradients here!
def evaluate(loader, network, criterion):
    network.eval()  # put network into evaluation mode (mostly for batch normalization)
    losses = []
    accuracies = []
    for inputs, labels in loader:
        inputs = inputs.to(device)  # put inputs and labels on GPU (if it is available)
        labels = labels.to(device)
        outputs = network(inputs)  # pass inputs through network to get outputs
        loss = criterion(outputs, labels)  # evaluate outputs with criterion to get loss
        accuracy = (torch.max(outputs, dim=1)[1] == labels).to(torch.float32).mean()  # accuracy
        losses.append(loss.cpu().numpy())
        accuracies.append(accuracy.cpu().numpy())
    return np.mean(losses), np.mean(accuracies)


epoch_max = 32  # you can increase this if you want (takes longer, but accuracy will be higher)
results = {"train losses": [], "train accuracies": [], "test losses": [], "test accuracies": []}

for epoch in tqdm(range(epoch_max)):
    for inputs, labels in tqdm(train_loader, leave=False):
        inputs = inputs.to(device)  # put inputs and labels on GPU (if it is available)
        labels = labels.to(device)
        inputs = data_augmentation(inputs)  # perform data augmentation on inputs
        network.train()  # put network into training mode (mostly for batch normalization)
        optimizer.zero_grad()  # zero-out gradients
        outputs = network(inputs)  # pass inputs through network to get outputs
        loss = criterion(outputs, labels)  # evaluate outputs bwith criterion to get loss
        loss.backward()  # backpropagate through loss to compute gradients (loss w.r.t. weights)
        optimizer.step()  # use gradients to perform optimization (e.g., NAdam)

    # after every epoch, evaluate results on the entire training & testing set (slow)
    train_loss, train_accuracy = evaluate(train_loader, network, criterion)
    test_loss, test_accuracy = evaluate(test_loader, network, criterion)
    # store results
    results["train losses"].append(train_loss)
    results["train accuracies"].append(train_accuracy)
    results["test losses"].append(test_loss)
    results["test accuracies"].append(test_accuracy)
    # print results so we can see how the network is doing
    result = f"{('[' + str(epoch + 1) + ']'):5s}   " \
             f"Train: {str(train_accuracy * 100):.6}% ({str(train_loss):.6})   " \
             f"Test: {str(test_accuracy * 100):.6}% ({str(test_loss):.6})"
    tqdm.write(result)

epochs = np.arange(1, epoch_max + 1)
# plot losses
plt.figure(figsize=(10, 7))
plt.plot(epochs, results["train losses"], label="train")
plt.plot(epochs, results["test losses"], label="test")
plt.title("Network loss (lower is better)", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plot accuracies
plt.figure(figsize=(10, 7))
plt.plot(epochs, np.asarray(results["train accuracies"]) * 100, label="train")
plt.plot(epochs, np.asarray(results["test accuracies"]) * 100, label="test")
plt.title("Network accuracy (higher is better)", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# show plots
plt.show()

# you can put whatever extension you want, .pth (or .pt) is just the PyTorch convention
torch.save(network.state_dict(), "cifar10_model.pth")