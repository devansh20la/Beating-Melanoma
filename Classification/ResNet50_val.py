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
from model import network
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interpolate

def save_checkpoint(state, is_best, filename,savetrainloss,savetraincorrects,savevalloss,savevalcorrects):
    torch.save(state, filename)

    savetrainloss_name = open('trainloss' + str(epoch) + '.pkl','wb')
    savetraincorrects_name = open('traincorrects' + str(epoch) + '.pkl','wb')
    savevalloss_name = open('valloss' + str(epoch) + '.pkl','wb')
    savevalcorrects_name = open('valcorrects' + str(epoch) + '.pkl','wb')

    pickle.dump(savetrainloss,savetrainloss_name)
    pickle.dump(savetraincorrects,savetraincorrects_name)
    pickle.dump(savevalloss,savevalloss_name)
    pickle.dump(savevalcorrects,savevalcorrects_name)

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
    if phase == 'train':
        outputs = outputs[0]	
    loss = criterion(outputs, labels)
    _, preds = torch.max(outputs.data, 1)
    
    if phase=='train':
        loss.backward()
        optimizer.step()

    corrects = torch.sum(preds==labels.data)
    
    return loss.data[0],corrects,outputs

def cal_auc_roc(submission):
    y = label_binarize(submission[:,0],classes=[0,1,2])
    n_classes = y.shape[1]
    y_score = submission[:,1:]
    y_pred = np.argmax(y_score,1)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i]) 
    
    fpr["micro"],tpr["micro"],_ = roc_curve(y.ravel(),y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"],tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
         mean_tpr += interp(all_fpr, fpr[i],tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])    
    return roc_auc


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
args = parser.parse_args()

print("learning_rate: {0}, decay:{1}, checkpoint:{2}".format(args.lr,args.lr_de,args.checkpoint))

data_transforms = {
    'train': transforms.Compose([
        # transforms.Scale(400),
        RandomVerticleFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Scale(400),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        # transforms.Scale(400),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}
print ("....Initializing data sampler.....")
data_dir = os.path.expanduser('~/Documents/Melanoma/Classification/data')
dsets = {x: imageandlabel(os.path.join(data_dir, x),'img_'+ x +'.csv', data_transforms[x])
         for x in ['train', 'val','test']}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=25, num_workers=10, shuffle=True) 
                for x in ['train', 'val','test']}

print ("....Loading Model.....")
model_ft = network()

if torch.cuda.is_available():
    model_ft = model_ft.cuda()

print ("....Model loaded....")
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
# optimizer = optim.Adam(model_ft.parameters(),lr=args.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd)

if args.checkpoint:
    state = torch.load(args.checkpoint)
    shutil.copyfile(args.checkpoint,'prev_' + args.checkpoint)
    model_ft.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    best_loss = state['best_loss']
    #best_acc = -float('inf')
    print("checkpoint Loaded star_epoch = {0},best_loss= {1}".format(start_epoch,best_loss))
    del state
    
    savetrainloss_name = open('trainloss' + str(start_epoch) + '.pkl','rb')
    savetraincorrects_name = open('traincorrects' + str(start_epoch) + '.pkl','rb')
    savevalloss_name = open('valloss' + str(start_epoch) + '.pkl','rb')
    savevalcorrects_name = open('valcorrects' + str(start_epoch) + '.pkl','rb')

    savetrainloss = pickle.load(savetrainloss_name)
    savetraincorrects = pickle.load(savetraincorrects_name)
    savevalloss = pickle.load(savevalloss_name)
    savevalcorrects = pickle.load(savevalcorrects_name)

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

    testloss = 0.0
    testcorrects = 0.0
    submission = np.zeros((600,4),dtype=float)

    for i,testdata in enumerate(dset_loaders['test'],0):
        tsinput, tslabels = testdata['image'],testdata['label']
        loss,correct,output = model_run('val',model_ft,tsinput,tslabels, criterion, optimizer)
        testloss +=  loss
        testcorrects += correct
        submission[25*i:25*i+25,0] = tslabels.numpy()
        submission[25*i:25*i+25,1:] = output.cpu().data.numpy()

    auc_roc = cal_auc_roc(submission)
        
    testloss = testloss/(i+1)
    testcorrects = testcorrects/(i+1)
   
    savevalloss[epoch] = valloss
    savevalcorrects[epoch] = valcorrects
 
    if testloss < best_loss:
        best_loss = testloss
        is_best = 1
    else:
        is_best = 0

    save_checkpoint({'epoch': epoch,
    'model': model_ft.state_dict(),
    'optimizer': optimizer.state_dict(),
    'best_loss': best_loss},is_best,'checkpoint_ep%d.pth.tar'%(epoch),savetrainloss,savetraincorrects,savevalloss,savevalcorrects)
    
    print ('Epoch = {0}, TrainingLoss = {1}, Train_corrects = {3},val Loss = {2}, val_corrects{4},testloss = {5},testcorrects = {6}, auc_roc = {7}'.format(epoch,trainloss,valloss,traincorrects,valcorrects,testloss,testcorrects,auc_roc["macro"]))
