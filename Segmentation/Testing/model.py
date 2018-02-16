import torch.nn as nn
import torchvision
from torchvision import transforms
from data_loader import seg_loader
import torch
import torch.optim as optim
from torch.autograd import Variable
import os
import numpy as np
import torchvision.utils as vutils
import math 


# Down-sampling block in the Unet Architecture 
class down_block(nn.Module):
	#using the input channels I specify the channels at for repeated use of this block
	def __init__(self,channels):
		super(down_block,self).__init__()
		self.conv1 = nn.Conv2d(channels[0],channels[1],kernel_size=(3,3),stride=1,padding=0,dilation=1,bias=True)
		self.conv2 = nn.Conv2d(channels[1],channels[2],kernel_size=(3,3),stride=1,padding=0,dilation=1,bias=True)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=(2,2),stride=2)

	#forward function through the block
	def forward(self,x):
		fwd_map = self.conv1(x)
		fwd_map = self.relu(fwd_map)
		fwd_map = self.conv2(fwd_map)
		# Saving the tensor to map it to the layers deeper in the model
		fwd_map = self.relu(fwd_map)
		out = self.maxpool(fwd_map)
		return (fwd_map,out)

# Up-sampling block in the U-Net Architecture 
class up_block(nn.Module):

	def __init__(self,channels):
		super(up_block,self).__init__()
		self.upsample = nn.Upsample(scale_factor=2,mode='nearest')
		self.up_conv = nn.Conv2d(channels[0],channels[1],kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.ConvTranspose2d(channels[0],channels[2],kernel_size=(3,3),stride=1,padding=0,dilation=1,bias=True)
		self.conv2 = nn.ConvTranspose2d(channels[2],channels[3],kernel_size=(3,3),stride=1,padding=0,dilation=1,bias=True)

	def forward(self,x,prev_x):
		out = self.upsample(x)
		out = self.up_conv(out)
		self.relu(out)
		#Asserting that the size of feed forward tensor is same as the one generated
		assert out.size(2) == prev_x.size(2)

		out = torch.cat((out,prev_x),dim=1)
		out = self.conv1(out)
		self.relu(out)
		out = self.conv2(out)
		self.relu(out)
		return out


#Combining both the down-sampling and up-sampling block into one network.
class network(nn.Module):

	def __init__(self):
		super(network,self).__init__()
		self.layer1 = down_block((7,64,64))
		self.layer2 = down_block((64,128,128))
		self.layer3 = down_block((128,256,256))
		self.layer4 = nn.Sequential(nn.Conv2d(256,512,kernel_size=(3,3),stride=1,padding=0,dilation=2,bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(512,512,kernel_size=(3,3),stride=1,padding=0,dilation=2,bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(512,1024,kernel_size=(3,3),stride=1,padding=0,dilation=4,bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(1024,1024,kernel_size=(3,3),stride=1,padding=0,dilation=4,bias=False),
			nn.ReLU(inplace=True))
		self.layer5 = nn.Sequential(nn.ConvTranspose2d(1024,1024,kernel_size=(3,3),stride=1,padding=0,dilation=4,bias=False),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(1024,512,kernel_size=(3,3),stride=1,padding=0,dilation=4,bias=False),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(512,512,kernel_size=(3,3),stride=1,padding=0,dilation=2,bias=False),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(512,512,kernel_size=(3,3),stride=1,padding=0,dilation=2,bias=False),
			nn.ReLU(inplace=True))

		self.layer6 = up_block((512,256,256,256))
		self.layer7 = up_block((256,128,128,128))
		self.layer8 = up_block((128,64,64,64))
		self.seg_map = nn.Conv2d(64,1,kernel_size=(1,1),stride=1,bias=False)


		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))

	def forward(self,x):
		fw_out1,out = self.layer1(x)
		fw_out2,out = self.layer2(out)
		fw_out3,out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)

		out = self.layer6(out,fw_out3)
		out = self.layer7(out,fw_out2)
		out = self.layer8(out,fw_out1)
		
		out = self.seg_map(out)

		return out

def center_crop(x, target_size):
    batch_size, n_channels, layer_width, layer_height = x.size()
    xy1 = (layer_width - target_size) // 2
    return x[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    
