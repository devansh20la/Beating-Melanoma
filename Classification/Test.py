import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision import datasets,models,transforms
import os
import copy
from data_loader import imageandlabel
import pdb
from model import network


def model_run(model,inputs,labels,criterion):
        
    model.eval()
    model.train(False)

    inputs, labels, = Variable(inputs.cuda(), volatile = True), Variable(labels.cuda(), volatile= True)
    outputs = model(inputs)

    loss = criterion(outputs, labels)
    _, preds = torch.max(outputs.data, 1)
    corrects = torch.sum(preds==labels.data)
   
    return loss.data[0],corrects,outputs

data_transforms = transforms.Compose([
        transforms.Scale(400),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data_dir = '~/Documents/Melanoma/Classification/data/test/'
dsets = imageandlabel(os.path.expanduser(data_dir),'img_test.csv', data_transforms)
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=25, shuffle=True, num_workers=10)

criterion = nn.CrossEntropyLoss()
#criterion = nn.BCEWithLogitsLoss()
i=0
submission = {}
model_ft = models.resnet50()
model_ft.fc = nn.Linear(2048,3)
model_ft.load_state_dict(torch.load('model_best.pth.tar')['model'])
model_ft.cuda()

for data in dset_loaders:
    trinput,trlabels,path = data['image'],data['label'],data['path']
    loss,correct,output = model_run(model_ft, trinput, trlabels, criterion)
    submission[i] = (loss,correct,trlabels,output.cpu(),path)
    i+=1

np.save('submission.npy',submission)
