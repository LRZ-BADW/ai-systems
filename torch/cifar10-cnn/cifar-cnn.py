# Import packages
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Hyper-parameters
num_epochs = 20
batch_size = 128
learning_rate = 0.001
image_width = 32
image_channels = 3
conv1_out_channels = 50
conv2_out_channels = 75
kernel_size = 5
pool_size = 2
fc1_out_channels = 50
num_classes = 10
dim_1 = int((image_width-kernel_size+1)/pool_size)
dim_2 = int((dim_1-kernel_size+1)/pool_size)

# Build model
class Model(nn.Module):
    def __init__(self, image_width, image_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(image_channels, conv1_out_channels, kernel_size)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool_size)
        self.fc1 = nn.Linear(conv2_out_channels*(dim_2**2), fc1_out_channels)
        self.fc2 = nn.Linear(fc1_out_channels, num_classes)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# download the data
train_dataset = torchvision.datasets.CIFAR10(root='../../data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root='../../data', train=False, transform=transforms.ToTensor(), download=True)

# transform to DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Training and testing function for which we only change the device type to cpu or gpu.
def train_and_test_CNN(device):
    print("Device: ", device)
    # Transfer the model to the chosen device
    model = Model(image_width, image_channels, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Data type
    dtype = torch.float

    # Additional information for checkpointing
    PATH = "/workspace/torch/cifar10-cnn/cnn_model.pt"

    # Load the checkpoint
    if(os.path.exists(PATH)):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Number of epochs from the checkpoint(s): ", checkpoint['epoch'])

    # Start time recording
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()

    # Train the model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            # Forward pass and compute loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print loss
            if (i+1) % 390 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{(i+1)*len(images)}/{len(train_dataset)}], Loss: {loss.item():.4f}')
        if (epoch+1) % 5 == 0:
            # Update checkpointing variables
            if (os.path.exists(PATH)):
                checkpoint = torch.load(PATH)
                EPOCH = checkpoint['epoch'] + 5
            else:
                EPOCH = epoch+1
            # Save the checkpoint
            torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, PATH)


    # Test the model - In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Test set: Accuracy: {acc} %')

    # End time recording
    ender.record()

    # Elapsed time
    time = starter.elapsed_time(ender)
    print(f'Elapsed time for training and testing: {time/1000} s \n\n')



###### Part 1: Train on a CPU ######

# Device configuration - cpu
#device = torch.device('cpu')
# Call train and test function
#train_and_test_CNN(device)


###### Part 2: Train on a GPU ######

# Device configuration - gpu/cuda
device = torch.device('cuda')
# Call train and test function
train_and_test_CNN(device)