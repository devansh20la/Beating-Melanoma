import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision import datasets,models,transforms
import os
import copy
from data_loader import imageandlabel, RandomVerticleFlip
import pandas as pd
import random
import shutil
import argparse
import pickle
import numpy as np

def save_checkpoint(state, is_best, filename):

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        print("New Best Model Found")

def model_run(phase,model,inputs,labels,criterion,optimizer):
        
    if phase == 'train':
        model.train()
    else:
        model.eval()

    if phase == 'train':
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs.cuda(), volatile = True), Variable(labels.cuda(), volatile= True)

    optimizer.zero_grad()
    outputs = model(inputs)

    loss = criterion(outputs, labels)
    _, preds = torch.max(outputs.data, 1)
    
    if phase=='train':
        loss.backward()
        optimizer.step()

    corrects = torch.sum(preds==labels.data)
    
    return loss.data[0],corrects,outputs

manualSeed = 200
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(200)

if torch.cuda.is_available():
   torch.cuda.manual_seed_all(manualSeed)

parser = argparse.ArgumentParser(description='PyTorch Skin Lesion Training')
parser.add_argument('--lr','--learning_rate',type=float,default=0.001,help='initial learning rate')
parser.add_argument('--lr_de','--lr_decay',type=int,default=30,help='learning rate decay epoch')
parser.add_argument('--checkpoint',type=str,default='')
parser.add_argument('--wd','--weightdecay',type=float,default=0)
parser.add_argument('--rd','--root_dir',default='/home/devansh/Documents/Melanoma/Classification/data')
parser.add_argument('--bs','--batch_size',type=int,default=50)

args = parser.parse_args()

print("learning_rate: {0}, decay:{1}, checkpoint:{2}".format(args.lr,args.lr_de,args.checkpoint))

data_transforms = {
    'train': transforms.Compose([
        RandomVerticleFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}
print ("....Initializing data sampler.....")

data_dir = args.rd
dsets = {x: imageandlabel(os.path.join(data_dir, x),'img_'+ x +'.csv', data_transforms[x])
         for x in ['train', 'val']}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.bs, num_workers=10, shuffle=True) 
                for x in ['train', 'val']}

print ("....Loading Model.....")
model_ft = models.resnet50(pretrained=True)
model_ft.fc = nn.Linear(2048,3)

if torch.cuda.is_available():
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
# optimizer = optim.Adam(model_ft.parameters(),lr=args.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd)

if args.checkpoint:
    state = torch.load(args.checkpoint)
    model_ft.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    best_loss = state['best_loss']
    savetrainloss = state['train_loss']
    savetraincorrects = state['train_corrects']
    savevalloss = state['val_loss']
    savevalcorrects = state['val_corrects']

    print("checkpoint Loaded star_epoch = {0},best_loss= {1}".format(start_epoch,best_loss))
    del state

    start_epoch+=1

else:
    start_epoch = 0
    best_loss = float('inf')
    savetrainloss = {}
    savetraincorrects = {}
    savevalloss = {}
    savevalcorrects = {}

for epoch in range(start_epoch,500):
    
    trainloss = 0.0
    traincorrects = 0.0

    for i,data in enumerate(dset_loaders['train'],1):
        trinput,trlabels = data['image'],data['label']
        loss,correct,_ = model_run('train',model_ft, trinput, trlabels, criterion, optimizer)
        trainloss += loss
        traincorrects += correct

    trainloss = trainloss/i
    traincorrects = traincorrects/i

    savetrainloss[epoch] = trainloss
    savetraincorrects[epoch] = traincorrects

    valloss = 0.0
    valcorrects = 0.0

    for i,valdata in enumerate(dset_loaders['val'],1):
        tsinput, tslabels = valdata['image'],valdata['label']
        loss,correct,output = model_run('val',model_ft,tsinput,tslabels, criterion, optimizer)
        valloss +=  loss
        valcorrects += correct

    valloss = valloss/i
    valcorrects = valcorrects/i
   
    savevalloss[epoch] = valloss
    savevalcorrects[epoch] = valcorrects
 
    if valloss < best_loss:
        best_loss = valloss
        is_best = 1
    else:
        is_best = 0

    save_checkpoint({'epoch': epoch,
    'model': model_ft.state_dict(),
    'optimizer': optimizer.state_dict(),
    'best_loss': best_loss,'train_loss':savetrainloss,'train_corrects':savetraincorrects,'val_loss':savevalloss,'val_corrects':savevalcorrects},is_best,'checkpoints/checkpoint_ep%d.pth.tar'%(epoch))
    
    print ('Epoch = {0}, TrainingLoss = {1}, Train_corrects = {3},val Loss = {2}, val_corrects{4}'.format(epoch,trainloss,valloss,traincorrects/args.bs,valcorrects/args.bs))
