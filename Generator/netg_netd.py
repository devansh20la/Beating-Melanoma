import torch
import torch.nn as nn

class Generator(nn.Module):	
	def __init__(self,inplanes=100):
		super(Generator,self).__init__()
		self.model = nn.Sequential(
			
			# input = batch_sizex10x1x1
			nn.ConvTranspose2d(10,256,kernel_size=4,stride=1,padding=0,bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			# Output = batch_sizex512x4x4

			nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1,bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			# Output = batch_sizex256x8x8

			nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1,bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			# Output = batch_sizex128x16x16

			nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1,bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			# Output = batch_sizex64x32x32

			nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=1,bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(True),
			#output = batch_sizex32x64x64

			nn.ConvTranspose2d(16,8,kernel_size=4,stride=2,padding=1,bias=False),
			nn.BatchNorm2d(8),
			nn.ReLU(True),
			#output = batch_sizex16x128x128
			
			nn.ConvTranspose2d(8,3,kernel_size=4,stride=2,padding=1,bias=False),
			nn.Tanh(),
			# Output = batch_sizex3x256x256

			)

	def forward(self,inp):
		output = self.model(inp)
		return output

class Descriminator(nn.Module):

	def __init__(self,inplanes=3):
		super(Descriminator,self).__init__()
		self.model = nn.Sequential(
			#Input = batch_sizex3x256x256
			nn.Conv2d(3,16,kernel_size=4,stride=2,padding=1,bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			#Output = batch_sizex16x128x128

			nn.Conv2d(16,32,kernel_size=4,stride=2,padding=1,bias=False),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2, inplace=True),
			#output = batch_sizex32x64x64

			nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1,bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, inplace=True),
			
			#output = batch_sizex64x32x32
			nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
			#Output = batch_sizex128x16x16

			nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),
			#Output = batch_sizex256x8x8

			nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1,bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True),
			#Output = batch_sizex512x4x4

			nn.Conv2d(512,1,kernel_size=4,stride=1,padding=0,bias=False),
			nn.Sigmoid()
			)

	def forward(self,inp):
		output = self.model(inp)
		return output
