import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms

transfor = transforms.Scale(400)

class imageandlabel(Dataset):

    def __init__ (self,root_dir,csv_file,trans=None):
        self.csvfile = pd.read_csv(csv_file)
        self.csvfile.sort_values('label',axis=0,ascending=True,kind='quicksort',inplace=True)
        self.csvfile.reset_index(inplace=True,drop = True)
        self.trans = trans
        self.root_dir = root_dir

    def __len__(self):
        return len(self.csvfile)

    def __getitem__(self,idx):
        imgname = os.path.join(self.root_dir,self.csvfile.iloc[idx,0])
        image = Image.open(imgname + '.jpg')
        label = self.csvfile.ix[idx,1]

        if image.size[0] != 256:
            image = transfor(image)

        if self.trans:
            image = self.trans(image)

        sample = {'image': image, 'label': label, 'path': imgname}
        return sample

class RandomVerticleFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


