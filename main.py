""" system """
import importlib
import os
import sys

""" Torch, Augmentations and PL  """
import torch
from torchvision import datasets, transforms as T, models
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn import functional as F
from torch.optim import lr_scheduler 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import ColorJitter, Blur, GaussNoise, GaussianBlur, ToGray
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

""" Imports from this directory """
from shapebias.helper.imagenet_dataset import ImageNetDataset
from shapebias.helper.auxiliary import get_accuracy
from shapebias.models.basic import BasicNetwork
from shapebias.helper import *
from shapebias.evaluate_models import ShapeBiasEvaluator
from shapebias.cue_conflict_dataset import CueConflictDataloader

""" others """
from pympler import asizeof
import wandb
import argparse


""" Helper class """
#TODO move this to the helper directory
class Transform_Dataset(Dataset):
    """
    This is so that we can take a dataset and append it a transformation.
    It is particularly useful when we want to use data augmentations to train that 
    are different from those to validate.
    """
    def __init__(self,dataset,transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        image,label = self.dataset[idx]
        if self.transform:
            if isinstance(self.transform,A.Compose): #Albumentations librar
                augmented = self.transform(image=image)
                image = augmented['image']
            else: 
                image = self.transform(image)
        return image, label

   
""" Callback/Hook for Torch Lightning """
#TODO move this to its special directory
class EvaluationHook(Callback):
    """
    This is a hook that is called on the start of the validation
    epoch and it calculates the accuracy of the model on the validation set
    and the shape-texture bias
    """
    def __init__(self,val_dataset,train_dataset):
        super().__init__()
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        path = os.path.dirname(os.path.abspath(__file__)) + "/shapebias/cue-conflict"
        self.shapebias_evaluator = ShapeBiasEvaluator(path)

    def on_validation_epoch_start(self, trainer, model):
        print("calculating accuracies... ")
        assert not model.training
        val_acc = get_accuracy(model,self.val_dataset,10)
        train_acc = get_accuracy(model,self.train_dataset,10)
        results = self.shapebias_evaluator(model)
        model.log('val-accuracy',val_acc)
        model.log('1000-train_accuracy',train_acc)
        model.log('shape-bias',results["shape_bias"])
        model.log('shape-match',results["shape_match"])
        model.log('texture-match',results["texture_match"])
        model.log('learning-rate',model.optimizer.param_groups[0]["lr"])
        print("\nAccuracy Validation: " + str(val_acc))
        print("Accuracy Training: " + str(train_acc))
        print("Shape_Bias: " + str(results["shape_bias"]))
        print("Learning Rate: " + str(model.optimizer.param_groups[0]["lr"]))

    

""" Main """
def main(configuration_module):
    """ Init Configuration """
    config = configuration_module.config
    wandb_logger = wandb.init(config=config,name='noise_blur_color',project='Imagenet_Shapebias')
    wandb_logger.save(configuration_module.__file__)

    """ End Configurations """
    
    config["gpus"] = [1]

    """ Training """
    trainer = pl.Trainer(
                gpus=config["gpus"],
                callbacks=[EvaluationHook(configuration_module.val_dataset,configuration_module.train_dataset)],
                logger= WandbLogger(experiment=wandb_logger,save_dir='logs'))   
    trainer.fit(configuration_module.model, configuration_module.train_loader, configuration_module.val_loader)



if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Introduce the configuration module at configurations directory (for example \"simple_imagenet\")")
    configuration_module = importlib.import_module("configurations."+sys.argv[1])
    main(configuration_module)



