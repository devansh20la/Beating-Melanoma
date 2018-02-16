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
from model import network
from sklearn import metrics
import shutil 
import random
import argparse
from scipy import ndimage
import pickle
from data_loader import RandomVerticalFlip

torch.backends.cudnn.benchmark=True

def save_checkpoint(state, train_loss, val_loss, is_best, filename,epoch):
    torch.save(state, filename)

    train_loss_name = open('train_loss' + str(epoch) + '.pkl','wb')
    val_loss_name = open('val_loss' + str(epoch) + '.pkl','wb')

    pickle.dump(train_loss,train_loss_name,pickle.HIGHEST_PROTOCOL)
    pickle.dump(val_loss,val_loss_name,pickle.HIGHEST_PROTOCOL)

    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        print("Best model found")

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

	output_map = output_map.data.numpy()
	seg_map = seg_map.data.cpu().numpy()

	j_index = 0.0
	for i in range(size):
		j_index += metrics.jaccard_similarity_score(seg_map[i,:,:].astype(int),output_map[i,:,:].astype(int),normalize=True)
	#print (j_index/size)

	return j_index/size

def lr_sch(optimizer, i, init_lr, lr_decay_batch):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr * (0.1**(i // lr_decay_batch))
        if i % lr_decay_batch == 0:
                 print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        return optimizer


# running the model and updated the loss based on the mini-batch. Additionally, phase input helps to separate the training and testing phase of 
# the model
def model_run(model,phase,input_img,seg_map,opt,cri):

	if phase == 'val':
		model.eval()
	else:
		model.train()
	output = model(input_img)

	loss = cri(output,seg_map)
	opt.zero_grad()

	if phase == 'train':
		loss.backward()
		opt.step()
	
	output[output>=0.4]=1
	output[output<0.4]=0
	return (output,loss)

manualSeed = 500
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(500)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(manualSeed)

parser = argparse.ArgumentParser(description='PyTorch Skin Lesion Training')
parser.add_argument('--lr','--learning_rate',type=float,default=1e-4,help='initial learning rate')
parser.add_argument('--lr_de','--lr_decay',type=int,default=30,help='learning rate decay epoch')
parser.add_argument('--checkpoint',type=str,default='')
parser.add_argument('--wd','--weight_decay',type=int,default=0)
args = parser.parse_args()

# Creating data tranform model for both training and validation
data_trans = {'train': transforms.Compose([transforms.ToTensor()]),
		'val':transforms.Compose([transforms.ToTensor()])}

root_dir = '/home/devansh/Documents/Melanoma/Segmentation/data/'

print (".....Setting up the data loader and sampler.....")

# Setting the data loader and data queueing processes. 
dsets = {x: seg_loader(root_dir + x,trans = data_trans[x]) for x in ['train', 'val']}
data_loader = {x: torch.utils.data.DataLoader(dsets[x], batch_size=15, shuffle=True, num_workers=5) for x in ['train', 'val']}
criterion = nn.BCEWithLogitsLoss()

print("....Initializing the model......")
model = network()

if torch.cuda.is_available():
	model = model.cuda()
	criterion = criterion.cuda()
	model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(),lr = args.lr,betas=(0.9, 0.999), eps=1e-08,weight_decay=args.wd)

if args.checkpoint:
    state = torch.load(args.checkpoint)
    shutil.copyfile(args.checkpoint,'prev' + args.checkpoint)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    best_loss = state['best_loss']
    print("checkpoint Loaded epoch = {0},best_loss = {1}".format(start_epoch,best_loss))
    del state
    train_loss_name = open('train_loss' + str(start_epoch) + '.pkl','rb')
    val_loss_name = open('val_loss' + str(start_epoch) + '.pkl','rb')
    save_train_loss = pickle.load(train_loss_name)
    save_val_loss = pickle.load(val_loss_name)
    start_epoch+=1

else:
    start_epoch = 0
    best_loss = float('inf')
    save_val_loss = {}
    save_train_loss = {}

print (".....Training the model....")
for epoch in range(start_epoch,500):

	# Training the model, saving the average loss across the entire epoch in train_loss
	optimizer = lr_sch(optimizer,epoch-start_epoch,args.lr,args.lr_de)

	train_loss = 0.0
	j_train = 0.0

	for idx,data in enumerate(data_loader['train'],1):
		#print(idx)
		input_img,input_img_hsv,input_L,seg_map = data['image'],data['image_hsv'],data['image_L'],data['segmentation']

		input_img = Normalize(input_img,(0.5,0.5,0.5),(0.5,0.5,0.5))
		input_img_hsv = Normalize(input_img_hsv,(0.5,0.5,0.5),(0.5,0.5,0.5))
		input_L = Normalize(input_L,(0.5),(0.5))

		input_img = torch.cat((input_img,input_img_hsv,input_L),dim=1)

		del input_img_hsv,input_L

		if torch.cuda.is_available():
			input_img, seg_map = Variable(input_img.cuda(),requires_grad=False), Variable(seg_map.cuda(),requires_grad=False)
		else:
			input_img, seg_map = Variable(input_img,requires_grad=False), Variable(seg_map,requires_grad=False)

		out_map,loss = model_run(model,'train',input_img,seg_map,optimizer,criterion)
		out_map = out_map.cpu()

		for i in range(out_map.size()[0]):
			out_map.data[i,0,:,:] = torch.from_numpy(ndimage.binary_fill_holes(out_map[i,0,:,:].data.numpy()).astype(int))
		
		train_loss += loss.data[0]
		j_train += jaccard(out_map,seg_map)
		#print (j_train)

		if idx%1 == 0:
			vutils.save_image(input_img.data[:,:3,:,:],'results/train_epoch%ditr%d_a.png'%(epoch,idx),normalize=True) 
			vutils.save_image(out_map.data,'results/train_epoch%ditr%d_c.png'%(epoch,idx))
			vutils.save_image(seg_map.data,'results/train_epoch%ditr%d_b.png'%(epoch,idx))

	train_loss = train_loss/idx
	j_train = j_train/idx
	#print (j_train)

	# Validating the model on the validation dataset, saving the average validation loss in val_loss
	val_loss = 0.0
	j_val = 0.0

	for idx,data in enumerate(data_loader['val'],1):
		#print(idx)
		input_img,input_img_hsv,input_L,seg_map = data['image'],data['image_hsv'],data['image_L'],data['segmentation']
		
		input_img = Normalize(input_img,(0.5,0.5,0.5),(0.5,0.5,0.5))
		input_img_hsv = Normalize(input_img_hsv,(0.5,0.5,0.5),(0.5,0.5,0.5))
		input_L = Normalize(input_L,(0.5),(0.5))
		
		input_img = torch.cat((input_img,input_img_hsv,input_L),dim=1)

		del input_img_hsv,input_L

		if torch.cuda.is_available():
			input_img, seg_map = Variable(input_img.cuda(),volatile=True), Variable(seg_map.cuda(),volatile=True)
		else:
			input_img, seg_map = Variable(input_img,volatile=True), Variable(seg_map,volatile=True)

		out_map,loss = model_run(model,'val',input_img,seg_map,optimizer,criterion)
		out_map = out_map.cpu()

		for i in range(out_map.size()[0]):
			out_map.data[i,0,:,:] = torch.from_numpy(ndimage.binary_fill_holes(out_map[i,0,:,:].data.numpy()).astype(int))
		
		j_val += jaccard(out_map,seg_map)
		#print(j_val)
		val_loss += loss.data[0]

		if idx%50 == 0:
			vutils.save_image(input_img.data[:,:3,:,:],'results/val_epochg%ditr%d_a.jpg'%(epoch,idx),normalize=True) 
			vutils.save_image(out_map.data,'results/val_epoch%ditr%d_c.jpg'%(epoch,idx))
			vutils.save_image(seg_map.data,'results/val_epoch%ditr%d_b.jpg'%(epoch,idx))

	val_loss = val_loss/idx
	j_val = j_val/idx
	#print(j_val)

	# Appending and saving the losses to an array for easy visualization while the model is still training.
	save_train_loss[epoch] = train_loss
	save_val_loss[epoch] = val_loss

	# Printing the average loss on the terminal
	print ("Epoch: {0}/500, Train_Loss: {1},J_train:{2}, Val_Loss: {3}, J_val: {4}".format(epoch,train_loss,j_train,val_loss,j_val))

	# Saving the model if the validation loss is less than the previous validation losses.
	if val_loss < best_loss:
		is_best = 1
		best_loss = val_loss
	else:
		is_best = 0

	save_checkpoint({'epoch': epoch, 
		'model':model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'best_loss': best_loss},save_train_loss,save_val_loss,is_best,'checkpoint_ep%d.pth.tar'%(epoch),epoch)


