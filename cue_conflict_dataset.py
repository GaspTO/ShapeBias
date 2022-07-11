import torch
from torchvision import transforms
import torchvision.datasets as datasets

class CueConflictDataset(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder"""

    def __init__(self, root_path,transform=None):
        super(CueConflictDataset, self).__init__(root_path,transform=transform)

    def __getitem__(self, index):
        """override the __getitem__ method. This is the method that dataloader calls."""
        sample, _ = super().__getitem__(index)

        path = self.imgs[index][0]
        file_name = path.split("/")[-1][:-4]
        raw_shape = file_name.split("-")[0]
        shape = ''.join([i for i in raw_shape if not i.isdigit()])
        raw_texture = file_name.split("-")[1][:-1]
        texture = ''.join([i for i in raw_texture if not i.isdigit()])

        assert shape in self.class_to_idx
        assert texture in self.class_to_idx
        
        return sample,shape,texture,path
    

class CueConflictDataloader(object):
    """Pytorch Data loader"""

    def __call__(self, root_path, resize, batch_size, num_workers):
        """
        Data loader for pytorch models
        :param path:
        :param resize:
        :param batch_size:
        :param num_workers:
        :return:
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if resize:
            transformations = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transformations = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])


        self.dataset = CueConflictDataset(root_path, transformations)
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        return loader
    
    @property
    def classes(self) -> list:
        return self.dataset.classes
    
    @property
    def class_to_idx(self) -> dict:
        return self.dataset.class_to_idx
    
    @property
    def imgs(self) -> list:
        return self.dataset.imgs
    
    