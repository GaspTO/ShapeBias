import torch
from torch.nn import functional as F

""" Helper functions """
def freeze_all(model):
    for params in model.parameters():
            params.requires_grad = False
    return model

def freeze_all_but_fc(model):
    for params in model.parameters():
            params.requires_grad = False
    for params in model.fc.parameters(): 
            params.requires_grad = True
    return model

def unfreeze_all(model):
    for params in model.parameters():
            params.requres_grad = True
    return model     

def reset_fc(model):
    model.fc = type(model.fc)(model.fc.in_features,model.fc.out_features)
    for params in model.fc.parameters():
            assert params.requires_grad
    return model
                
def check_if_frozen_but_fc(model):
    for params in list(model.parameters())[:-2]:
            assert not params.requires_grad

def unormalize(images_batch,device):
    """
    Image net models normalize the images based on pixel 
    averages. This code unormalizes them. 
    This might be useful if we want to plot/watch the image.
    """
    x = images_batch

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

