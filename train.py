""" Fine-tuning last layer models """
from torchvision import datasets, transforms as T, models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import ColorJitter, Blur, GaussNoise, GaussianBlur
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.nn import functional as F
from os.path import abspath, dirname
from torch.utils.data import random_split, DataLoader

""" Imports from torch core """
import sys
sys.path.insert(0, "/home/guests2/tro/torch-core")
from imagenet_dataset import ImageNetDataset
from auxiliary import get_accuracy, get_crossentropy
from imagenet_light import Model as Good_Model

""" Imports from this directory """
from evaluate_models import ShapeBiasEvaluator
from cue_conflict_dataset import CueConflictDataloader


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


""" Helper functions """
def freeze_all(model):
        for params in model.parameters():
                params.requires_grad = False
        return model


def freeze_all_but_fc(model):
        for params in model.parameters():
                params.requires_grad = False
        for params in model.fc.parameters(): #!
                params.requires_grad = True
        return model

def unfreeze_all(model):
        for params in model.parameters():
                params.requres_grad = True
        return model     

def reset_fc(model):
        #model.fc = type(model.fc)(model.fc.in_features,model.fc.out_features)
        model.fc = type(model.fc)(model.fc.in_features,model.fc.out_features)
        for params in model.fc.parameters():
                assert params.requires_grad
        return model
                
def check_if_frozen_but_fc(model):
    #except last
    for params in list(model.parameters())[:-2]:
            assert not params.requires_grad

def unormalize(batch,device):
    x = batch

    mean1 = torch.ones(1,x[0].shape[1],x[0].shape[2]) * 0.485
    mean2 = torch.ones(1,x[0].shape[1],x[0].shape[2]) * 0.456
    mean3 = torch.ones(1,x[0].shape[1],x[0].shape[2]) * 0.406
    mean = torch.concat((mean1,mean2,mean3),dim=0) * 255.0


    std1 = torch.ones(1,x[0].shape[1],x[0].shape[2]) * 0.229
    std2 = torch.ones(1,x[0].shape[1],x[0].shape[2]) * 0.224
    std3 = torch.ones(1,x[0].shape[1],x[0].shape[2]) * 0.225
    std = torch.concat((std1,std2,std3),dim=0) * 255.0

    new_x = x[0] * std.to(device) + mean.to(device)
    new_x = new_x.unsqueeze(0)

    return new_x



""" Model """

class Modelo(pl.LightningModule):
    def __init__(self,pretrained=True): 
            super().__init__()
            self.model = models.resnet50(pretrained=pretrained)    
    
    def forward(self,x):
            embedding = self.model(x)
            return embedding

    def configure_optimizers(self):
            #self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4,weight_decay=1e-4)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
            return self.optimizer

        
    def training_step(self,train_batch,batch_idx):
            assert self.model.training
            #check_if_frozen_but_fc(self.model)
            x, y = train_batch
            x = x.float()
            z = self.model(x)    
            loss = F.cross_entropy(z, y)
            self.log('train_loss', loss)


            #self.experiment.image(0, x)
            tensorboard = self.logger.experiment
            x = unormalize(x,torch.device("cuda"))
            tensorboard.add_images('new_images',x.to(torch.uint8),batch_idx)

            #tensorboard.add_images('images',x[0].to(torch.uint8).unsqueeze(0),batch_idx)
            #tensorboard.add_images('new_images',new_x.to(torch.uint8),batch_idx)
            #print(x[0].dtype)
            #print(type(tensorboard))
            return loss

    def validation_step(self,val_batch,batch_idx):
            assert not self.model.training
            #check_if_frozen_but_fc(self.model)
            x, y = val_batch
            z = self.model(x)
            loss = F.cross_entropy(z, y)
            self.log('val_loss', loss)
     
    def freeze_all_but_fc(self):
        freeze_all_but_fc(self.model)
            
    def unfreeze(self):
        unfreeze_all(self.model)

    def reset_fc(self):
        reset_fc(self.model)
    

""" Callback/Hook for Torch Lightning """
class MyEvaluation(Callback):
    def __init__(self,val_dataset,train_dataset):
        super().__init__()
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        self.shapebias_evaluator = ShapeBiasEvaluator("cue-conflict")

    def on_validation_epoch_start(self, trainer, model):
        print("calculating accuracies... ")
        assert not model.training
        val_acc = get_accuracy(model,self.val_dataset,10)
        train_acc = get_accuracy(model,self.train_dataset,10)
        results, info = self.shapebias_evaluator.evaluate(model,verbose=False)
        model.log('val-accuracy',val_acc)
        model.log('1000-train_accuracy',train_acc)
        model.log('shape-bias',results["shape_bias"])
        model.log('shape-match',results["shape_match"])
        model.log('texture-match',results["texture_match"])
        print("\nAccuracy Validation: " + str(val_acc))
        print("Accuracy Training: " + str(train_acc))
        print(info)

    

""" Main """
def main():
    """ Initializations """
    model = Modelo()

    """ Transformations """
    null_transform = A.Compose([A.Resize(224, 224),
                                ToTensorV2()])

    standard_transform = T.Compose([
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), 
                            T.Resize((224,224))]) 

    standard_transform = A.Compose([
                                    A.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225],
                                    ),
                                    A.Resize(224, 224), 
                                    ToTensorV2()
                                ])


    special_transform = A.Compose([
                                A.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                ),
                                A.Resize(224, 224),
                                A.Compose([
                                        ColorJitter(p=1),
                                        GaussianBlur(p=1),
                                        GaussNoise(p=1),
                                ],p=0.5),
                                ToTensorV2()
    ])

    gaussnoise_transform = A.Compose([
                                A.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                ),
                                A.Resize(224, 224),
                                GaussNoise(var_limit=(0.0,0.005),p=1), #!
                                ToTensorV2()
    ])


    color_jitter_transform = A.Compose([
                                A.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                ),
                                A.Resize(224, 224),
                                ColorJitter(p=1), #!
                                ToTensorV2()
    ])




    

    print("using special transform")
    dataset = ImageNetDataset()
    print(len(dataset))
            

    """ Prepare model and check its accuracy"""
    model.reset_fc() 
    model.freeze_all_but_fc() 
    model.eval()
    #acc = get_accuracy(model,Transform_Dataset(dataset,transform=standard_transform),1000) #!
    acc = get_accuracy(model,Transform_Dataset(dataset,transform=standard_transform),10)
    print("accuracy:"+str(acc)) #should be pretty low, if not 0.0
    assert acc < 0.10


    """ Train"""
    total_data_percentage, train_val_percentage = 0.2, 0.9
    split_dataset, garbage = random_split(dataset, [int(len(dataset)*total_data_percentage), len(dataset)-int(len(dataset)*total_data_percentage)])

    train_data, val_data = random_split(split_dataset, [int(len(split_dataset)*train_val_percentage), len(split_dataset)-int(len(split_dataset)*train_val_percentage)])
    
    train_data = Transform_Dataset(train_data,transform=gaussnoise_transform) #!
    val_data = Transform_Dataset(val_data,transform=standard_transform)

    print("size of train dataset used: "+str(len(train_data)))
    print("size of val dataset used: "+str(len(val_data)))
    #train_loader = DataLoader(train_data, batch_size=64,num_workers=10,persistent_workers=True) #!
    #val_loader = DataLoader(val_data, batch_size=64,num_workers=10,persistent_workers=True) #!
    train_loader = DataLoader(train_data, batch_size=64,num_workers=10,persistent_workers=True,shuffle=True) #!
    val_loader = DataLoader(val_data, batch_size=64,num_workers=10,persistent_workers=True,shuffle=False) #!

    #logger = TensorBoardLogger('logs', name='test')
    trainer = pl.Trainer(gpus=[0],callbacks=[MyEvaluation(val_data,train_data)])
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()


 #gaussian blur 0.5 Default resnet18 ZEZERE --- killed not work
 
 #gaussian blur 1  Default resnet18 ZEZERE --- kileed not work

 #gaussian blur 0 Default resnet18 TEJO -- killed works well
 
 #gaussian blur 1 (0,2) resnet18 Zezere -- killed not work

 #gaussian blur 1 (0,0.5) resnet18 Zezere -- killed not work

 #gaussian blur 1 (0,0.05) resnet18 Zezere 

 #color jitter Default resnet18 Tejo -- killed not work

 #gaussian blur 1 (0,0.005) resnet18 Zezere

 #gaussian blur 1 (0,0.005) resnet50 tejo
