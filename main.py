import os
import sys

""" Fine-tuning last layer models """
from torchvision import datasets, transforms as T, models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import ColorJitter, Blur, GaussNoise, GaussianBlur, ToGray
from torch.utils.data import Dataset, random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from torch.nn import functional as F

""" Imports from torch core """
from shapebias.helper.imagenet_dataset import ImageNetDataset
from shapebias.helper.auxiliary import get_accuracy
from shapebias.models.basic import BasicNetwork
from shapebias.helper import *

""" Imports from this directory """
from shapebias.evaluate_models import ShapeBiasEvaluator
from shapebias.cue_conflict_dataset import CueConflictDataloader

""" debugg """
from pympler import asizeof


""" Helper class """
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
def main():
    """ Transformations """ 
    train_transform = T.Compose([
            T.ToTensor(),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
    val_transform = T.Compose([
            T.ToTensor(),
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])


    """ Initializations """
    net = models.resnet18(False)
    model = BasicNetwork(net)
    train_dataset = ImageNetDataset(img_dir="/media/imagenet/train/",transform=train_transform)
    val_dataset = ImageNetDataset(img_dir="/media/imagenet/val/",transform=val_transform)  
    print("train dataset length: " + str(len(train_dataset)))
    print("validation dataset length: " + str(len(val_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=256,num_workers=10,persistent_workers=True,shuffle=True) #!
    val_loader = DataLoader(val_dataset, batch_size=1024,num_workers=10,persistent_workers=True,shuffle=False) #!

    trainer = pl.Trainer(
                gpus=[0],
                callbacks=[EvaluationHook(val_dataset,train_dataset)],
                )   
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()


