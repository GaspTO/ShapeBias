"""Fine-tuning last layer models"""


from torchvision import datasets, transforms as T, models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from torch.nn import functional as F
import sys
from os.path import abspath, dirname
from torch.utils.data import random_split, DataLoader

""" Imports from torch core """
sys.path.insert(0, "/home/guests2/tro/torch-core")
from imagenet_dataset import ImageNetDataset
from auxiliary import get_accuracy, get_crossentropy
from imagenet_light import Model as Good_Model




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
	#except last
	for params in list(model.parameters())[:-2]:
			assert not params.requires_grad








class Resnet18(pl.LightningModule):
	def __init__(self,pretrained=True):
			super().__init__()
			self.model = models.resnet18(pretrained=pretrained)	
	
	def forward(self,x):
			embedding = self.model(x)
			return embedding

	def configure_optimizers(self):
			self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
			return self.optimizer

		
	def training_step(self,train_batch,batch_idx):
			assert self.model.training
			#check_if_frozen_but_fc(self.model)
			x, y = train_batch
			z = self.model(x)    
			loss = F.cross_entropy(z, y)
			self.log('train_loss', loss)
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
	


class MyEvaluation(Callback):
	def __init__(self,dataset):
		super().__init__()
		self.dataset = dataset

	def on_validation_epoch_start(self, trainer, model):
		acc = get_accuracy(model,self.dataset,100)
		model.log('100-accuracy',acc)
		print("Validating Externally " + str(acc))

	
	""" 
	Just to test
	def on_train_batch_start(self,trainer,model,batch,batch_idx):
		assert model.training
		model.eval()
		assert not model.training
		acc = get_accuracy(model,self.dataset,100)
		model.log('100-accuracy',acc)
		print("batch "+str(batch_idx) + " Externally " + str(acc))
		model.train()
		assert model.training
	"""



def main():
	""" Initializations """
	model = Resnet18(True)
	standard_transform = T.Compose([
											T.ToTensor(),
											T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), 
											T.Resize((224,224))]) 
	dataset = ImageNetDataset(transform=standard_transform)
	print(len(dataset))
			



	""" Prepare model and check its accuracy"""
	model.reset_fc()
	model.freeze_all_but_fc()
	model.eval()
	acc = get_accuracy(model,dataset,100)
	print("accuracy:"+str(acc)) #should be pretty low, if not 0.0
	assert acc < 0.10


	""" Train"""
	total_data_percentage, train_val_percentage = 0.4, 0.8
	split_dataset, garbage = random_split(dataset, [int(len(dataset)*total_data_percentage), len(dataset)-int(len(dataset)*total_data_percentage)])
	train_data, val_data = random_split(split_dataset, [int(len(split_dataset)*train_val_percentage), len(split_dataset)-int(len(split_dataset)*train_val_percentage)])
	print("size of dataset used: "+str(len(train_data)))
	train_loader = DataLoader(train_data, batch_size=128,num_workers=10,persistent_workers=True) #!
	val_loader = DataLoader(val_data, batch_size=128,num_workers=10,persistent_workers=True) #!

	trainer = pl.Trainer(gpus=[0],callbacks=[MyEvaluation(dataset)])
	trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
	main()