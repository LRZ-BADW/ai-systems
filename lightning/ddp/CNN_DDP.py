#!/usr/bin/env python
# coding: utf-8

# Import packages
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchmetrics import Accuracy
import pytorch_lightning as pl
import time
from pytorch_lightning.loggers import TensorBoardLogger
import logging

class cnnModel(pl.LightningModule):
    
    def __init__(self, model, learning_rate, num_classes):
        
        super().__init__()
        
        self.save_hyperparameters()
                
        if model=="resnet":
            self.model = models.resnet34(num_classes=num_classes)
            
        elif model=="vgg19":
            self.model = models.vgg19(num_classes=num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        
    def forward(self,x):
        return self.model(x)
    
    def training_step(self, train_batch, batch_idx):
        
        imgs, labels = train_batch
        preds = self.forward(imgs)
        loss = self.criterion(preds, labels)
        self.log("train_loss", loss)
        
        acc = self.accuracy(preds.argmax(dim=-1), labels)
        self.log("Train Acc", acc, on_step=False, on_epoch=True)
        
        return loss    
        
    def validation_step(self, val_batch, batch_idx):
        
        imgs, labels = val_batch
        preds = self.forward(imgs)
        loss = self.criterion(preds, labels)
        self.log("val_loss", loss)
        
        acc = self.accuracy(preds.argmax(dim=-1), labels)
        self.log("Val Acc", acc, on_step=False, on_epoch=True)
        
    def test_step(self, test_batch, batch_idx):
        
        imgs, labels = test_batch
        preds = self.forward(imgs)
        loss = self.criterion(preds, labels)
        self.log('Test loss', loss)
        
        acc = self.accuracy(preds.argmax(dim=-1), labels)
        self.log("Test Acc", acc, on_step=False, on_epoch=True)
        
    def configure_optimizers(self):
        
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)


def runCNNmodel(modelName='resnet', gpus=0):
    
    # Hyper-parameters
    num_epochs = 5
    batch_size = 128
    learning_rate = 0.001
    num_classes = 10

    # configure logging at the root level of Lightning
    logging.getLogger("lightning").setLevel(logging.ERROR)

    torch.manual_seed(42)
    
    # download the data
    train_dataset = torchvision.datasets.CIFAR10(root='/workspace/data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='/workspace/data', train=False, transform=transforms.ToTensor(), download=True)

    # transform to DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model = cnnModel(modelName, learning_rate, num_classes)
    
    logger = TensorBoardLogger("/workspace/lightning/ddp/logs", name=modelName+"_ddp")
    logger = None
    accelerator = 'gpu'
    devices = gpus
    
    print("Training {} with {} gpus - DDP...\n".format(modelName, gpus))

    trainer = pl.Trainer(max_epochs=num_epochs,
                         enable_progress_bar=False,
                         logger=logger,
                         devices=devices,
                         accelerator=accelerator,
                         strategy='ddp',
                         deterministic="warn")
        
    timeA = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    print("Time for training {} on {}: ".format(modelName, accelerator), time.time()-timeA)
    
    trainer.test(model, dataloaders=test_loader)

if __name__=="__main__":
    
    runCNNmodel(modelName='vgg19', gpus=2)