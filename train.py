"""Fine-tuning last layer models"""


import torchvision.models as models

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
        
        
model = models.resnet18()
freeze_all(model)
reset_fc(model)
for params in model.fc.parameters():
        assert params.requires_grad
unfreeze_all(model)



import ligthning

class model(lightning)
	def __init__(self,model):
		self.model = model
		
	
	def forward(x):
		return self.model(x)


        def validatation_step(self,x):
		pass
		

	def train_step(self,x):
		pass
		

	def freeze_layers_except_fc(self):
		pass	
	
	

        
trainer = Trainer()
trainer.fit()

