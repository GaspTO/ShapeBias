import torch
from torchvision import transforms as T, models
from torch.optim import lr_scheduler 
import albumentations as A
from albumentations.augmentations import Normalize, HorizontalFlip, RandomResizedCrop, ColorJitter, Blur, GaussNoise, GaussianBlur, ToGray
from albumentations.pytorch import ToTensorV2
from shapebias.models.basic import BasicNetwork
from shapebias.helper.imagenet_dataset import ImageNetDataset
from torch.utils.data import Dataset, random_split, DataLoader

train_transform = A.Compose([
        RandomResizedCrop(224,224),
        ColorJitter(p=0.5),
        GaussianBlur(p=0.5),
        GaussNoise(p=0.5),
        HorizontalFlip(p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ToTensorV2() #tensor has to come in the end
    ])

val_transform = T.Compose([
        T.ToTensor(),
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])


net = models.resnet18(False)
optimizer = torch.optim.SGD(net.parameters(),0.1, weight_decay=1e-4,momentum=0.9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
batch_size = 256
shuffle_train_data = True

model = BasicNetwork(net,optimizer=optimizer,scheduler=scheduler)

train_dataset = ImageNetDataset(img_dir="/media/imagenet/train/",transform=train_transform)
val_dataset = ImageNetDataset(img_dir="/media/imagenet/val/",transform=val_transform)  
train_loader = DataLoader(train_dataset, 
                    batch_size=batch_size,
                    num_workers=10,
                    persistent_workers=True,
                    shuffle=shuffle_train_data) 
val_loader = DataLoader(val_dataset,
                    batch_size=1024,
                    num_workers=10,
                    persistent_workers=True,
                    shuffle=False) 


config = dict(
    train_transform = train_transform,
    val_transform = val_transform,
    dataset = "Imagenet",
    train_dataset_size = len(train_dataset),
    val_dataset_size = len(val_dataset),
    net = "pretrained resnet-18 from pytorchvision",
    optimizer = optimizer,
    scheduler = "reduce LR on Plateau",
    batch_size = batch_size,
    shuffle_train_data = shuffle_train_data,
)

