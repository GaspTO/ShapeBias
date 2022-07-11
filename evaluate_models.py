
""" 
STEP 1

Get torch model
pass it through input
get real labels
measure accuracy

"""

"""
STEP 2
use datafra,e

"""

from nbformat import write
from decision_mappings import ImageNetProbabilitiesTo1000ClassNamesMapping, ImageNetProbabilitiesTo16ClassNamesMapping
from helper import wordnet_functions as wnf

 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
import torchvision.datasets as datasets
import torch
from cue_conflict_dataset import CueConflictDataloader
import csv
from tqdm import tqdm


""" 
EVALUATION
"""
#dataset = ImageFolder("./cue-conflict",transform=transforms.ToTensor())
model = models.resnet101(pretrained=True)
#simple_mapp = ImageNetProbabilitiesTo1000ClassNamesMapping()
dataloader = CueConflictDataloader()("cue-conflict",True,10,4)
file_name = "results.csv"
max_lines = -1

def run(dataloader,model,filename="results.csv",max_lines=None):
    if max_lines == None: max_lines = -1
    mapp = ImageNetProbabilitiesTo16ClassNamesMapping()
    with open(filename,"w") as fp:
        writer = csv.writer(fp)
        writer.writerow(["response", "shape","texture", "imagename"])
        
        for x,shapes,textures,paths in tqdm(dataloader):
            with torch.no_grad():
                classes_batch = mapp(torch.nn.functional.softmax(model(x),dim=1).numpy())
                for classes,shape,texture,path in zip(classes_batch,shapes,textures,paths):
                    writer.writerow([classes[0],shape,texture,path])
                    max_lines -= 1
                    if max_lines == 0: break
            if max_lines == 0: break


""" 
PLOT
"""
def plot(filename="results.csv",verbose=True):
    total,shape_or_texture, shape, texture = 0,0,0,0
    with open(filename,"r") as fp:
        reader = csv.reader(fp)
        reader.__next__() #dump first line
        for row in reader:
            if row[0] == row[1]:
                shape_or_texture += 1
                shape += 1
            elif row[0] == row[2]:
                shape_or_texture += 1
                texture += 1
            total += 1
        info = ""
        info += "shape: " + str(shape) + "\n"
        info += "texture: " + str(texture) + "\n"
        info += "shape or texture: " + str(shape_or_texture) + "\n"
        info += "total: " + str(total) + "\n"
        info += "shape%: " + str(shape/shape_or_texture) + "\n"
        info += "texture%: " + str(texture/shape_or_texture) + "\n"
        if verbose: print(info)
        return info


def evaluate(dataloader,model,filename="results.csv",verbose=True):
    evaluate(dataloader,model,filename=filename)
    info = plot(filename,verbose)
    return info





