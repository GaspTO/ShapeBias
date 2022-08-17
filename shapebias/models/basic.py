#from ..helper.fun import unormalize, freeze_all_but_fc, unfreeze_all, reset_fc
from shapebias.helper import fun
import torch
from torchvision import models
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim import lr_scheduler as scheduler

""" Model """
class BasicNetwork(pl.LightningModule):
    def __init__(self,model,optimizer=None): 
        super().__init__() 
        self.model = model
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(),0.1, weight_decay=1e-4,momentum=0.9)
        self.scheduler = scheduler.ReduceLROnPlateau(self.optimizer)

    def forward(self,x):
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, 
                "lr_scheduler": self.scheduler,
                "monitor": "train_loss"}


    def training_step(self,train_batch,batch_idx):
        assert self.model.training
        #check_if_frozen_but_fc(self.model)
        x, y = train_batch
        x = x.float()
        z = self.model(x)    
        loss = F.cross_entropy(z, y)
        self.log('train_loss', loss)
        ''' 
        tensorboard = self.logger.experiment
        x = fun.unormalize(x,torch.device("cuda"))
        tensorboard.add_images('new_images',x.to(torch.uint8),batch_idx)
        '''
        return loss

    def validation_step(self,val_batch,batch_idx):
        assert not self.model.training
        x, y = val_batch
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        self.log('val_loss', loss)
     
    def freeze_all_but_fc(self):
        fun.freeze_all_but_fc(self.model)
            
    def unfreeze(self):
        fun.unfreeze_all(self.model)

    def reset_fc(self):
        fun.reset_fc(self.model)

    