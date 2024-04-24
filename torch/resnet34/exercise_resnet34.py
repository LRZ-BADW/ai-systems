# Import packages
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


# Hyper-parameters
num_epochs = 20
batch_size = 128
learning_rate = 0.001


# Training and testing function for which we only change the device type to cpu or gpu.
def train_and_test_CNN(device):
    print("Device: ", device)

    # Load the model
    model = models.resnet34(weights=None)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Download the data
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    # Transform to DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Data type
    dtype = torch.float

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
