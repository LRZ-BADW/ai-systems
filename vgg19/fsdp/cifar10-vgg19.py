import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from torchvision import models
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

def print_peak_memory(prefix, device):
    if device == 0:
        print("Max."+f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")
        print(f"{prefix}: {torch.cuda.memory_allocated(device) // 1e6}MB ")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_and_validate(model, trainloader, valloader, criterion, optimizer, rank, num_epochs):

    for epoch in range(num_epochs):
        # Training step
        if rank == 0:
            timeStart = time.time()

        trainloader.sampler.set_epoch(epoch)
        ddp_train_loss = torch.zeros(2).to(rank)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            if epoch==0:
                print_peak_memory("Memory allocated before loss backward()", rank)
            loss.backward()
            if epoch==0:
                print_peak_memory("Memory allocated before optimizer step()", rank)
            optimizer.step()
            if epoch==0:
                print_peak_memory("Memory allocated after optimizer step()", rank)
            ddp_train_loss[0] += loss.item()
            ddp_train_loss[1] += len(data)

        dist.all_reduce(ddp_train_loss, op=dist.ReduceOp.SUM)

        # Validation step
        model.eval()
        ddp_correct = 0
        ddp_total = 0
        valloader.sampler.set_epoch(epoch)
        ddp_val_loss = torch.zeros(2).to(rank)
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images, labels = images.to(rank), labels.to(rank)
                outputs = model(images)
                loss = criterion(outputs, labels)
                ddp_val_loss[0] += loss.item()
                ddp_val_loss[1] += len(data)
                _, predicted = torch.max(outputs.data, 1)
                ddp_total += labels.size(0)
                ddp_correct += (predicted == labels).sum().item()

        dist.all_reduce(ddp_val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(torch.tensor(ddp_correct).to(rank), op=dist.ReduceOp.SUM)
        dist.all_reduce(torch.tensor(ddp_total).to(rank), op=dist.ReduceOp.SUM)

        dist.barrier()

        if rank == 0:
            timeEnd = time.time()
            print('Epoch: %d, Time: %fs, Training Loss: %.3f, Validation Loss: %.3f, Validation Accuracy: %.3f %%' % \
                  (epoch + 1, timeEnd-timeStart, ddp_train_loss[0] / ddp_train_loss[1], ddp_val_loss[0] / ddp_val_loss[1], 100 * ddp_correct / ddp_total))
    model.train()

def main(rank, world_size, args):
    setup(rank, world_size)
    # Load CIFAR10 dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    download = True if rank == 0 else False

    trainset = torchvision.datasets.CIFAR10(root='/workspace/data', train=True, download=download, transform=transform)
    dist.barrier()

    # Split trainset into train and validation sets
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    # Create dataloaders for train and validation sets
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(valset, num_replicas=world_size, rank=rank)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, sampler=val_sampler)

    testset = torchvision.datasets.CIFAR10(root='/workspace/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    #set device
    torch.cuda.set_device(rank)

    # Load VGG19 model
    model = models.vgg19(num_classes=10)
    model = model.to(rank)
    print_peak_memory("Memory allocated after creating local model", rank)

    model = FSDP(model)
    print_peak_memory("Memory allocated after creating FSDP model", rank)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # Training
    init_start_event.record()
    train_and_validate(model, trainloader, valloader, criterion, optimizer, rank, args.epochs)

    init_end_event.record()

    if rank == 0:
        print(f"Total time for training and validating: {init_start_event.elapsed_time(init_end_event) / 1000}sec")

    # Testing
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(rank), labels.to(rank)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if rank == 0:
        print('Accuracy of the network on test images: %d %%' % (100 * correct / total))

    cleanup()

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='VGG19 Training')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    args = parser.parse_args()

    print("Parsed arguments:")
    print("Global Batch size:", args.batch_size)
    print("Number of epochs:", args.epochs)

    world_size = torch.cuda.device_count()
    args.batch_size = int(args.batch_size / world_size)
    print("Local (per-GPU) Batch size:", args.batch_size)

    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
