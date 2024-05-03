#!/usr/bin/env python
# coding: utf-8

# Import packages
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchmetrics import Accuracy
import lightning as pl
import time
from lightning.pytorch.loggers import TensorBoardLogger
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
from lightning.pytorch.callbacks import ModelCheckpoint

# Build model
class Model(pl.LightningModule):
    
    def __init__(self, image_width, image_channels, conv1_out_channels,
                 conv2_out_channels, kernel_size, pool_size, fc1_out_channels,
                 num_classes, dim_2, learning_rate):
        
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model architecture
        self.conv1 = nn.Conv2d(image_channels, conv1_out_channels, kernel_size)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool_size)
        self.fc1 = nn.Linear(conv2_out_channels*(dim_2**2), fc1_out_channels)
        self.fc2 = nn.Linear(fc1_out_channels, num_classes)
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def loss(self, outputs, labels):
        return self.criterion(outputs, labels)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
    def _calculate_loss(self, batch, mode="train"):
        
        imgs, labels = batch
        preds = self.forward(imgs)
        loss = self.loss(preds, labels)
        self.log('%s_loss' % mode, loss, on_step=True, on_epoch=True)
        
        acc = self.accuracy(preds.argmax(dim=-1), labels)
        self.log('%s_acc' % mode, acc, on_step=True, on_epoch=True)
        
        return loss, acc
        
        
    def training_step(self, train_batch, batch_idx):
        
        loss, _ = self._calculate_loss(train_batch, mode="train")
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        
        _ = self._calculate_loss(val_batch, mode="val")
    
    def test_step(self, test_batch, batch_idx):
        
        _ = self._calculate_loss(test_batch, mode="test")

        
def runCode(gpus=0):
    
    # Hyper-parameters
    num_epochs = 5
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
    
    torch.manual_seed(42)

    # download the data
    train_dataset = torchvision.datasets.CIFAR10(root='/workspace/data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='/workspace/data', train=False, transform=transforms.ToTensor(), download=True)

    # transform to DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    model = Model(image_width = image_width,
                  image_channels = image_channels,
                  conv1_out_channels = conv1_out_channels,
                  conv2_out_channels = conv2_out_channels,
                  kernel_size = kernel_size,
                  pool_size = pool_size,
                  fc1_out_channels = fc1_out_channels,
                  num_classes = num_classes,
                  dim_2 = dim_2,
                  learning_rate = learning_rate)
    
    modelName = "CNN"
    accelerator = 'cpu'
    devices = 'auto'
    if gpus > 0:
        modelName += "_gpu_{}".format(gpus)
        accelerator = 'gpu'
        devices = gpus
        
    logger = TensorBoardLogger("/workspace/lightning/cifar10-cnn/logs", name=modelName)
    
    checkpoint_callback = ModelCheckpoint(dirpath='/workspace/lightning/cifar10-cnn/saved_models/',
                                          filename=modelName+'-{epoch:02d}',
                                          
                                         )
    
    print("Training with {} gpus...\n".format(gpus))
    
    trainer = pl.Trainer(max_epochs=num_epochs,
                         logger=logger,
                         devices=devices,
                         accelerator=accelerator,
                         enable_progress_bar=False,
                         deterministic=True,
                         callbacks=[checkpoint_callback])
    
    timeA = time.time()

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    
    print("Time for training on {}: ".format(accelerator), time.time()-timeA)
    
    trainer.test(model, dataloaders=test_loader)

if __name__=="__main__":
    runCode(gpus=0)
    runCode(gpus=1)