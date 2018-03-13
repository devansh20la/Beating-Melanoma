import os
import torch 
from torchvision import datasets,models,transforms
import numpy as np 
from torch.autograd import Variable
from netg_netd import Generator, Descriminator
from torch import nn 
from PIL import Image
import torchvision.utils as vutils

def norm_ip(img,min,max):
	img.clamp_(min=min,max=max)
	img.add_(-min).div_(max-min)

trans = transforms.Compose([transforms.Scale(300),transforms.CenterCrop((256,256)),transforms.ToTensor()])
root_dir = 'Data/'
results_dir = 'results/'
data_dict = []

#Loading all the images
all_images = [fn for fn in os.listdir(root_dir) if fn.endswith('.jpg')]

# Load the generator model
print ("......Loading the model........")
model = Generator()
model.load_state_dict(torch.load('netG.pth'))

if torch.cuda.is_available():
	model.cuda()

results = {}
for epochs in range(10):
	#Generating a fixed noise with 0 mean and 1 std_deviation i.e what the generator was trained on
	fixed_noise = torch.FloatTensor(50, 10, 1, 1).normal_(mean=0, std=1)

	if torch.cuda.is_available():
		fixed_noise = Variable(fixed_noise.cuda(),volatile=True)
	else:
		fixed_noise = Variable(fixed_noise, volatile=True)

	# Generating output image
	output_image = model(fixed_noise)
	loss = nn.MSELoss()
#
	for idx,out_img in enumerate(output_image,0):
		norm_ip(out_img.data,min=out_img.data.min(),max=out_img.data.max())
		img_loss = float('inf')	
		for img_name in all_images:
			actual_img = Image.open(root_dir + img_name)
			actual_img = trans(actual_img)
			actual_img = torch.unsqueeze(actual_img,0)
			
			if torch.cuda.is_available():
				actual_img = actual_img.cuda()

			myloss = loss(actual_img,out_img).data[0]
			if myloss < img_loss:
				img_loss = myloss
				save_match_image = actual_img

		vutils.save_image(save_match_image,'results/image%d_m_%f.jpg'%(50*epochs + idx,img_loss),normalize=True)
		vutils.save_image(out_img.data,'results/image%d_m.jpg'%(50*epochs + idx),normalize=True)



