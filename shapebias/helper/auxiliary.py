import random
import torch
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from sklearn.neighbors import KNeighborsClassifier

def preparebatch(dataset,nums:list):
    x = []
    y = []
    if isinstance(nums,int): nums = [nums]
    for i in nums:
        x.append(dataset[i][0])
        y.append(dataset[i][1])
    x = torch.stack(x)
    y = torch.tensor(y)
    return x,y
    #x, y = dataset[i]
    #return [x.unsqueeze(0),torch.tensor([y])]

def run_model(model,dataset,i):
    #batch = preparebatch(dataset,i)
    x = dataset[i][0]
    if not isinstance(x,torch.Tensor):
        x = torch.tensor(x)
    x = x.unsqueeze(0).to(model.device)
    y = torch.nn.functional.softmax(model(x)).argmax()
    t = dataset[i][1]
    return int(y), int(t)
 
def get_accuracy(model,dataset,num_batches,batch_size=100):
    acc = 0
    n = num_batches 
    dataloader = DataLoader(dataset, batch_size=batch_size,num_workers=10,persistent_workers=True,shuffle=True)
    for x,y in dataloader:
        if n == 0:
            return acc/(num_batches*batch_size)
        t = torch.nn.functional.softmax(model(x.to(model.device)),dim=1).argmax(dim=1) #! x device is ... 
        #!t = model(x.to(model.device))
        results = (t == y.to(model.device))  #! y device
        acc += results[results==True].numel() 
        n -= 1

def get_crossentropy(model,dataset,num):
    total_loss = 0
    for i in range(num):
        batch = preparebatch(dataset,i)
        x = dataset[i][0].unsqueeze(0)
        z = model(x)
        t = dataset[i][1]
        total_loss += F.cross_entropy(z,torch.tensor([t]))
    return total_loss/num
