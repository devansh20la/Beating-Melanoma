import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms
from skimage import color
# Defining a class for data loader __len__ and __getitem__ are two required functions to be able to use it 
# with sampling and other pytorch classes.

class seg_loader(Dataset):

    def __init__ (self,root_dir,trans=None):
        self.trans = trans
        self.root_dir = root_dir
        self.files = [fn for fn in os.listdir(root_dir) if fn.endswith('.jpg')]

    def __len__(self):
        return len(self.files)

    # Open the image and the segmentation map and pass it through the tranforms, return path as well so that later
    # one could save the results as csv files with imagenames
    def __getitem__(self,idx):

        # reading and resizing the image and concatenating HSV 

        imgname = os.path.join(self.root_dir,self.files[idx])
        image = Image.open(imgname)
        image = image.resize((400,400))
        image = image.crop(box=(10,10,390,390))

        # Converting to HSV
        image_hsv = image.convert("HSV")

        #Converting to L
        image_L = color.rgb2lab(np.array(image))[:,:,0]
        image_L = Image.fromarray(image_L.astype('uint8'))

        # Reading and resizing the segmentation mask, make sure the aspect ratio remains same
        seg_name = os.path.join(self.root_dir,self.files[idx].split('.')[0])
        seg_image = Image.open(seg_name + '_segmentation.png')
        seg_image = seg_image.convert('L')
        seg_image = seg_image.resize((400,400))
        seg_image = seg_image.crop(box=(10,10,390,390))
        
        # data augmentation blocks 
        if random.random() < 0.5:
            image.transpose(Image.FLIP_LEFT_RIGHT)
            image_L.transpose(Image.FLIP_LEFT_RIGHT)
            image_hsv.transpose(Image.FLIP_LEFT_RIGHT)
            seg_image.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            image.transpose(Image.FLIP_TOP_BOTTOM)
            image_hsv.transpose(Image.FLIP_TOP_BOTTOM)
            image_L.transpose(Image.FLIP_TOP_BOTTOM)
            seg_image.transpose(Image.FLIP_TOP_BOTTOM)

        if self.trans:
            image = self.trans(image)
            image_L = self.trans(image_L)
            image_hsv = self.trans(image_hsv)
            seg_image = self.trans(seg_image)

        sample = {'image': image, 'image_hsv': image_hsv,'image_L':image_L, 'segmentation': seg_image, 'path': imgname}

        return sample

class RandomVerticalFlip(object):
	def __call__(self,img):
		if random.random() < 0.5:
			return img.transpose(Image.FLIP_TOP_BOTTOM)
		else: return img

