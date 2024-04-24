# Import packages
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import horovod.torch as hvd

# Hyper-parameters
num_epochs = 10
batch_size = 128
learning_rate = 0.001

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
if torch.cuda.is_available():
      torch.cuda.set_device(hvd.local_rank())

# Define datasets
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# Partition datasets among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())

# Define loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

# Define the model
model = models.resnet34(weights='IMAGENET1K_V1')
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: average metric values across workers.
def metric_average(val, name):
    tensor = val.clone().detach()
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

# Start time recording
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter.record()

# Train the model
for epoch in range(num_epochs):
   for batch_idx, (data, target) in enumerate(train_loader):
       data, target = data.cuda(), target.cuda()
       optimizer.zero_grad()
       output = model(data)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()
       if (batch_idx+1) % 195 == 0:
           # Horovod: use train_sampler to determine the number of examples in this worker's partition.
           print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, (batch_idx+1) * len(data), len(train_sampler), loss.item()))

# Test the model
with torch.no_grad():
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target)
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # Horovod: use test_sampler to determine the number of examples in this worker's partition.
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Accuracy: {:.2f}%\n'.format(100. * test_accuracy))

        # End time recording and print elapsed time
        ender.record()
        time = starter.elapsed_time(ender)
        print(f'Elapsed time for training and testing: {time/1000} s \n\n')
