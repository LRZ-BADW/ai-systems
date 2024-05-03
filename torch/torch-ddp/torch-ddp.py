import os
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Hyper-parameters
num_epochs = 10
batch_size = 128
learning_rate = 0.001


def train_and_test(gpu, gpus, nodes, nr, world_size):
    rank = nr * gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.manual_seed(0)

    # Define datasets
    train_dataset = torchvision.datasets.CIFAR10(root='../../data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='../../data', train=False, transform=transforms.ToTensor(), download=True)

    # Partition datasets among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    # Define loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=test_sampler)

    # Model
    model = models.vgg19(weights='IMAGENET1K_V1')
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Start time recording
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()

    # Train
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 195 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, (i + 1)*len(images), len(train_sampler), loss.item()))

    # Test
    with torch.no_grad():
        test_loss = 0.
        test_accuracy = 0.
        for images, labels in test_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            output = model(images)
            # sum up batch loss
            test_loss += criterion(output, labels)
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(labels.data.view_as(pred)).cpu().float().sum()

    if gpu == 0:
        test_accuracy /= len(test_sampler)
        print('\nTest set: Accuracy: {:.2f}%\n'.format(100. * test_accuracy))

        # End time recording
        ender.record()

        # Elapsed time
        time = starter.elapsed_time(ender)
        print(f'Elapsed time for training and testing: {time/1000} s \n\n')


if __name__ == '__main__':
    gpus = 2
    nodes = 1
    nr = 0 #ranking within the nodes
    world_size = gpus * nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    mp.spawn(train_and_test, nprocs=gpus, args=(gpus, nodes, nr, world_size))
