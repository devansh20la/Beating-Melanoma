import torch
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=16)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


A = np.load('submission.npy').item()
submission = np.zeros((600,4),dtype=float)
path = []


for i in range(0,len(A)):
	_,_,label,score,p = A[i]
	submission[25*i:25*i+25,0] = label.numpy()
	submission[25*i:25*i+25,1:] = score.data.numpy()
	path.append(p)

# Binarize the output
y = label_binarize(submission[:,0], classes=[0, 1, 2])
n_classes = y.shape[1]

y_score = submission[:,1:]
y_pred = np.argmax(y_score,1)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i]) 

fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# Plot all ROC curves
fig1 = plt.figure()

plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.3f})'
              ''.format(roc_auc["micro"]),
         color='red', linewidth=1,marker='s',markevery=5,markersize=3)

plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.3f})'''.format(roc_auc["macro"]),
         color='blue', linewidth=1,marker='*',markevery=5)


colors = cycle(['darkturquoise', 'green'])
for i, color, c in zip([0,2], colors, ['Melanoma','Seborrheic keratosis']):
   plt.plot(fpr[i], tpr[i], color=color, lw=2,label='ROC curve of {0} class (area = {1:0.3f})'''.format(c, roc_auc[i]))


plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title('Receiver operating characteristic')
legend = plt.legend(loc="lower right")
plt.setp(legend.get_texts(), color='black')

plt.tight_layout()


cnf_matrix = confusion_matrix(submission[:,0], y_pred,labels=[0,1,2])
fig = plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['M','N','SK'],
                     title='Confusion matrix')

plt.show()
