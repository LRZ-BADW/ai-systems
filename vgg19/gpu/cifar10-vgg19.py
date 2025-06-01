import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from torchvision import models
import time

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.memory_allocated(device) // 1e6}MB ")
        print("Max. "+f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

def train_and_validate(args, model, trainloader, valloader, criterion, optimizer, device, num_epochs):
    
    for epoch in range(num_epochs):
        # Training step
        train_loss = 0.0
        init_start_event = torch.cuda.Event(enable_timing=True)
        init_end_event = torch.cuda.Event(enable_timing=True)

        model.train()
        init_start_event.record()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.detach()
            if epoch==0 and i==0:
                print_peak_memory("Memory allocated before loss backward()", 0)
            loss.backward()
            if epoch==0 and i==0:
                print_peak_memory("Memory allocated after loss backward() and before optimizer step()", 0)
            optimizer.step()
            if epoch==0 and i==0:
                print_peak_memory("Memory allocated after optimizer step()", 0)

        init_end_event.record()
        torch.cuda.synchronize()
        step_time = init_start_event.elapsed_time(init_end_event)/1000
        images_per_sec = torch.tensor(len(trainloader)*args.batch_size/step_time).to(device)

        # Validation step
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.detach()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().detach()



        val_acc = 100 * correct / total

        print('Epoch: %d, Time: %f s, Images-per-sec: %f img/s, Training Loss: %.3f, Validation Loss: %.3f, Validation Accuracy: %.3f %%' % \
              (epoch + 1, step_time, images_per_sec, train_loss, val_loss, val_acc))

def main(args):

    # Load CIFAR10 dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/workspace/data', train=True, download=True, transform=transform)

    # Split trainset into train and validation sets
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    # Create dataloaders for train and validation sets
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='/workspace/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    #identify and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VGG19 model
    model = models.vgg19(num_classes=10)
    model = model.to(device)

    print_peak_memory("Memory allocated after creating local model", 0)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    timestart = time.time()
    train_and_validate(args, model, trainloader, valloader, criterion, optimizer, device, args.epochs)
    timeend = time.time()
    print("Total training time: ", timeend - timestart, "seconds")
    print("Training time per epoch: ", (timeend - timestart)/args.epochs, "seconds")

    # Testing
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    
    # Argument parser
    parser = argparse.ArgumentParser(description='VGG19 Training')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    args = parser.parse_args()

    print("Parsed arguments:")
    print("Batch size:", args.batch_size)
    print("Number of epochs:", args.epochs)

    main(args)
