import torch
from torchvision import transforms as T, models
from shapebias.models.basic import BasicNetwork
from torch.optim import lr_scheduler 

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

net = models.resnet18(False)
optimizer = torch.optim.SGD(net.parameters(),0.1, weight_decay=1e-4,momentum=0.9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
batch_size = 256
shuffle_train_data = True


config = dict(
    train_transform = train_transform,
    val_transform = val_transform,
    net = "pretrained resnet-18 from pytorchvision",
    optimizer = optimizer,
    scheduler = "reduce LR on Plateau",
    batch_size = batch_size,
    shuffle_train_data = shuffle_train_data
)
