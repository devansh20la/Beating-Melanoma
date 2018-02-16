import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
from data_loader import seg_loader
import torch
import torch.optim as optim
from torch.autograd import Variable
import os
import numpy as np
from PIL import Image
import pandas as pd 
from model import network 
torch.backends.cudnn.benchmark=True
from sklearn import metrics 

def Normalize(tensor,mean,std):

    for ten in tensor:
    	if ten.size(0) == 1:
    		ten.sub_(mean).div_(std)
    	else:
	        for t,m,s in zip(ten,mean,std):
	            t.sub_(m).div_(s)
    return tensor		
		

def jaccard(output_map,seg_map):
	size = output_map.size(0)

	output_map = torch.squeeze(output_map,1)
	seg_map = torch.squeeze(seg_map,1)

	output_map = output_map.data.cpu().numpy()
	seg_map = seg_map.data.cpu().numpy()

	j_index = 0.0
	for i in range(size):
		j_index += metrics.jaccard_similarity_score(seg_map[i,:,:].astype(int),output_map[i,:,:].astype(int),normalize=True)

	return j_index/size

# Creating data tranform model for both training and validation
data_trans = transforms.ToTensor()
root_dir = '/home/devansh/Documents/Melanoma/Classification/data/test/'

print (".....Initializing data loader and sampler.....")
# Setting the data loader and data queueing processes. 
dsets = seg_loader(root_dir,trans = data_trans)
data_loader = torch.utils.data.DataLoader(dsets, batch_size=1, shuffle=True, num_workers=2)
criterion = nn.BCEWithLogitsLoss()

print("....Initializing the model......")
model = network()
model = model.cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load('model_best.pth.tar')['model'])
model.eval()

if torch.cuda.is_available():
	criterion.cuda()

loss_list = []

print ("....Testing the model.....")
j_index = 0.0

for count,data in enumerate(data_loader,1):
	input_image,input_img_hsv,input_L,seg_map = data['image'],data['image_hsv'],data['image_L'],data['segmentation']
	input_image = Normalize(input_image,(0.5,0.5,0.5),(0.5,0.5,0.5))
	input_img_hsv = Normalize(input_img_hsv,(0.5,0.5,0.5),(0.5,0.5,0.5))
	input_L = Normalize(input_L,(0.5),(0.5))
	input_image = torch.cat((input_image,input_img_hsv,input_L),dim=1)

	del input_img_hsv

	if torch.cuda.is_available():
		input_image,seg_map = Variable(input_image.cuda(),volatile=True), Variable(seg_map.cuda(),volatile=True)
	else:
		input_image,seg_map = Variable(input_image,volatile=True), Variable(seg_map,volatile=True)


	out_map = model(input_image)
	loss = criterion(out_map,seg_map)
	out_map[out_map>=0.4]=1
	out_map[out_map<0.4]=0

	j_index += jaccard(out_map,seg_map)
	
	vutils.save_image(input_image.data[:,:3,:,:],'results/' + str(count) + '_a.eps',normalize=True)
	vutils.save_image(seg_map.data,'results/' + str(count) + '_b.eps',normalize=True)
	vutils.save_image(out_map.data,'results/' + str(count) + '_c.eps',normalize=True)
	loss_list.append((loss.data[0],j_index))

j_index = j_index/count
print (j_index)
loss_list = pd.DataFrame(data=loss_list)
loss_list.to_csv('my_file.csv')
