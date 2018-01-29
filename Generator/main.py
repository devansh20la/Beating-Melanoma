import torch
from netg_netd import Generator, Descriminator
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision import datasets,models,transforms
import os
import copy
import torchvision.utils as vutils
import random

batch_size = 50
channel_size = 10

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
	m.weight.data.normal_(1.0,0.02)

if torch.cuda.is_available():
	fixed_noise = Variable(torch.FloatTensor(batch_size, channel_size, 1, 1).normal_(mean=0, std=1).cuda())
else:
	fixed_noise = Variable(torch.FloatTensor(batch_size, channel_size, 1, 1).normal_(mean=0, std=1))

real_label = 1
fake_label = 0

Gen = Generator()
Gen_opt = optim.Adam(Gen.parameters(),lr=2e-4, betas=(0.5, 0.999))
Gen.apply(weights_init)

Des = Descriminator()
Des_opt = optim.Adam(Des.parameters(),lr=2e-4, betas=(0.5, 0.999))
Des.apply(weights_init)

criterion = nn.BCELoss()

data_transforms = transforms.Compose([transforms.Scale(300),
									transforms.CenterCrop((256,256)),
									transforms.ToTensor(),
									transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if torch.cuda.is_available():
	Gen = Gen.cuda()
	Des = Des.cuda()
	criterion = criterion.cuda()

dsets = datasets.ImageFolder(root='Data/', transform=data_transforms)
dataloader = torch.utils.data.DataLoader(dsets, batch_size=batch_size, shuffle=True, num_workers=10)
results = []

for epoch in range(1,500):
	for i, data in enumerate(dataloader,0):

		#.......Training Descriminator on real images..........
		#Des.train()
		#Gen.eval()
		
		real_img, _ = data
		
		Des_opt.zero_grad()

		real_img_label = torch.FloatTensor(real_img.size(0),1,1,1).fill_(real_label)

		if torch.cuda.is_available():
			real_img = Variable(real_img.cuda())
			real_img_label = Variable(real_img_label.cuda())
		else:
			real_img = Variable(real_img)
			real_img_label = Variable(real_img_label)


		output = Des(real_img)
		loss_D = criterion(output,real_img_label)
		loss_D.backward()
		pred_x = output.data.mean()

		#.......Training Descriminator on fake images..........
		noise = torch.FloatTensor(real_img.size(0),channel_size,1,1).normal_(mean=0,std=1)
		if torch.cuda.is_available():
			noise = Variable(noise.cuda())
		else:
			noise = Variable(noise)

		fake_img = Gen(noise)

		if torch.cuda.is_available():
			fake_img_label = Variable(torch.FloatTensor(real_img.size(0),1,1,1).fill_(fake_label).cuda())
		else:
			fake_img_label = Variable(torch.FloatTensor(real_img.size(0),1,1,1).fill_(fake_label))
			
		output = Des(fake_img.detach())
		loss = criterion(output,fake_img_label)
		loss.backward()
		loss_D = loss_D + loss
		pred_Gx = output.data.mean()

		Des_opt.step()

		#........Training Generator on fake images............
		#Des.eval()
		#Gen.train()

		Gen_opt.zero_grad()
		labels = torch.FloatTensor(real_img.size(0),1,1,1).fill_(real_label)
		
		if torch.cuda.is_available():
			labels = Variable(labels.cuda())
		else:
			labels = Variable(labels)

		output = Des(fake_img)
		pred_Gx_2 = output.data.mean()
		loss = criterion(output,labels)
		loss.backward()
		Gen_opt.step()
		loss_G = loss

		if epoch%10 == 0:
			for _ in range(2):			
				noise = torch.FloatTensor(real_img.size(0),channel_size,1,1).normal_(mean=0,std=1)
				if torch.cuda.is_available():
					noise = Variable(noise.cuda())
				else:
					noise = Variable(noise)

				fake_img = Gen(noise)
				Gen_opt.zero_grad()
				labels = torch.FloatTensor(real_img.size(0),1,1,1).fill_(real_label)
				
				if torch.cuda.is_available():
					labels = Variable(labels.cuda())
				else:
					labels = Variable(labels)

				output = Des(fake_img)
				pred_Gx_2 = output.data.mean()
				loss = criterion(output,labels)
				loss.backward()
				Gen_opt.step()

		print '[Epoch [{0}]/[500]],[iteration [{1}]/[{7}]], loss_Des = {2}, loss_Gen = {3}, pred_x = {4}, pred_G(x) = {5}, pred_G = {6}'.format(epoch,i,loss_D.data[0], loss_G.data[0], pred_x, pred_Gx, pred_Gx_2,len(dataloader))
		results.append((epoch,i,loss_D.data[0],loss_G.data[0],pred_x,pred_Gx))
		np.save('results.npy',results)
		if i % 10 == 0:
			#vutils.save_image(real_img.data,'images/real_samples_ep%d_it%d.png' % (epoch, i), normalize=True)
		#	Gen.eval()
			fake = Gen(fixed_noise)
			vutils.save_image(fake.data,'images/fake_samples_ep%d_it%d.png' % (epoch, i), normalize=True)
		
	torch.save(Gen.state_dict(), 'netG.pth')
	torch.save(Des.state_dict(), 'netD.pth')
	torch.save(Gen,'Gen.pth')













