
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

#from nbformat import write
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
from pl_bolts.models.self_supervised import SimCLR
import os



class ShapeBiasEvaluator:
    def __init__(self,dataset_path,resize:bool=True,batch_size=128,workers=4,device=None):
        self.dataset_path = dataset_path
        self.resize = resize
        self.batch_size = batch_size
        self.workers = workers
        self.dataloader = CueConflictDataloader()("cue-conflict",True,self.batch_size,self.workers)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def _run(self,model,filename="results.csv",max_lines=None):
        max_lines = max_lines if max_lines is not None else -1        
        mapp = ImageNetProbabilitiesTo16ClassNamesMapping()
        with open(filename,"w") as fp:
            writer = csv.writer(fp)
            writer.writerow(["response", "shape","texture", "imagename"])
            for x,shapes,textures,paths in tqdm(self.dataloader):
                with torch.no_grad():
                    x = x.to(self.device)
                    output = torch.nn.functional.softmax(model(x),dim=1)
                    classes_batch = mapp(output.cpu().numpy())
                    for classes,shape,texture,path in zip(classes_batch,shapes,textures,paths):
                        writer.writerow([classes[0],shape,texture,path])
                        max_lines -= 1
                        if max_lines == 0: break
                if max_lines == 0: break

    def _summarize(self,filename="results.csv",verbose=True):
        results = {}
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
            results = {"total":total,
                    "shape_bias":shape/shape_or_texture,
                    "shape_match":shape/total,
                    "texture_match":texture/total}
            info = ""
            info += "total: " + str(results["total"]) + "\n"
            info += "shape_bias: " + str(results["shape_bias"]) + "\n"
            info += "shape_match: " + str(results["shape_match"]) + "\n"
            info += "texture_match: " + str(results["texture_match"]) + "\n"
            if verbose: print(info)
            return results, info

    def evaluate(self,model,filename="results.csv",max_lines=None,verbose=True,keep_result_file=False):
        model.to(self.device)
        self._run(model,filename=filename)
        results, info = self._summarize(filename,verbose)
        assert os.path.exists(filename)
        os.remove(filename)
        return results, info

    



if __name__ == '__main__':
    import timm
    #model = models.resnet18(pretrained=True)
    model = timm.create_model('resnet18d',pretrained=True)
    dataloader = CueConflictDataloader()("cue-conflict",True,128,4)
    file_name = "results.csv"
    max_lines = -1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    verbose = True
    ShapeBiasEvaluator("cue-conflict").evaluate(model)



