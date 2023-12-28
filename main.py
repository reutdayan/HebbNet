import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from hebb_net import HebbNet, HebbRuleWithActivationThreshold
from model import test, train
from visualization import visualize_weights

# Hyperparameters
batch_size = 1
lr = 1 # η, the learning rate
p = 0.01  # top-p percentile for gradient sparsification
epochs = 200
momentum = 5e-4
lr_decay = 0.95

input_layer_size = 28*28
hidden_layer_size = 2000
output_layer_size = 10

# Define dataset
# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Load the model architecture
model = HebbNet(input_layer_size, hidden_layer_size, output_layer_size).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([
            {'params': model.classification_weights.parameters(), 'lr': lr, 'momentum': momentum} ])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1, gamma=lr_decay)
activation_thresholder = HebbRuleWithActivationThreshold().to(device)

for t in range(epochs):
    print(f"Epoch {t}\n-------------------------------")
    train(train_dataloader, model, device, p, loss_fn, optimizer, scheduler.get_last_lr()[0], activation_thresholder)
    test(test_dataloader, model, device, loss_fn)
    scheduler.step()
    torch.cuda.empty_cache()
print("Done!")

visualize_weights(model.hebbian_weights.weight)



