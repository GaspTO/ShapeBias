from torch.utils.data import Dataset
import os 
from PIL import Image
import numpy as np
import albumentations
from tqdm import tqdm


class ImageNetDataset(Dataset):
    def __init__(self, annotations_file="/media/imagenet/train/words.txt", img_dir='/media/imagenet/train/', transform=None, target_transform=None, max_num_classes=1000):
        if img_dir[-1] != '/': img_dir += "/"
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data_paths = []
        self.classes = []
        self.class_nums = {}
        fp = open(annotations_file)
        lines = fp.readlines()
        o = 0
        bar = tqdm(total=max_num_classes, unit="class")
        for line in lines:
            dir_name, class_name = line.replace('\n','').split('\t') #lines in words.txt
            path = img_dir + dir_name + '/'
            if os.path.isdir(path):
                for image in os.listdir(path):
                    image_path = path + image
                    label = self._add_class(class_name)
                    self.data_paths.append((image_path,label))
                bar.update()
                o += 1
        fp.close()
        assert len(self.data_paths) != 0, "Something went wrong. Dataset has 0 samples."     

    def _add_class(self,class_name):
        if class_name not in self.class_nums:
            self.class_nums[class_name] = len(self.classes)
            self.classes.append(class_name)
        return self.class_nums[class_name]
            
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.data_paths[idx][0]).convert('RGB'))
        label = self.data_paths[idx][1]
        if self.transform:
            if isinstance(self.transform,albumentations.Compose): #Albumentations library
                augmented = self.transform(image=image)
                image = augmented['image']
            else: 
                image = self.transform(image)
        
        if self.target_transform:
            raise Exception("Why would I transform the label?")
            label = self.target_transform(label)
        
        if image.shape[0] == 373:
            print("hey")
        return image, label



if __name__ == '__main__':
    import albumentations as A
    transform = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    dataset = ImageNetDataset(transform=transform)
    print(dataset[0])

