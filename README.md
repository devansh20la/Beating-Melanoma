# Beating Melanoma with Deep Learning: letting the data speak

The main focus of the reserach is to perform analysis of skin lesions with deep learning pipeline systems.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

```
Pytorch 0.2.0
```

## Hair removal

The hair removal is implemented using MATLAB.

Say what the step will be

```
1) Specify root directory for input images in main.m
2) Dermoscopic images can be super highresolution (7000x7000 or higher), resize images to get faster results. Resizing might require suitable adjustment to radius of structuring element and threshold value in my_function.
```

## Deep Convolutional Generative Adversarial Network (DCGAN)

###  Training:
The data loader class (data_loader.py) take csv file as input. The CSV file is arranged as follows:
```
Image_ID, Label
ISIC_0015284, 0
ISIC_0015284, 0
.
.
.
```

Fake images will be saved under images folder as fake_samples_ep%d_it%d.png  after every 10th batch of real data.  The model weights are saved as 'netG.pth' and 'netD.pth'.

```
usage main.py: Paste all the available data inside Generator/Data/Melanoma
```

### Mean squared error check:

```
usage mse_check.py: find closest match between fake images and real images.

1) The images are saved under results folder. Real closest matched images are saved as image%d_m_%f.jpg%(idx, error).
2)Generated images is saved as image%d_m.jpg %(idx).
```

## Lesion segmentation

Model weights can be downloaded from:

### Training

```
usage Unet.py: [--lr LEARNING_RATE] [--lr_de LEARNING_RATE_DECAY] [--checkpoint LOAD_PREV_CHECKPOINT] [-wd WEIGHT_DECAY] [-rd ROOT_DIR]

1) Checkpoints are saved at every epoch as checkpoint_ep%d.pth.tar %(epoch)
2) Validation and Training losses are saved at every epoch as train_loss%d.pkl %(epoch) and val_loss%d.pkl %(epoch).
3) Output images are save under results folder as:
i)  INPUT_IMAGE: val_epoch%ditr%d_a.jpg %(epoch,iteration)
ii) LABEL: val_epoch%ditr%d_b.jpg %(epoch,iteration)
iii) OUTPUT_MAP: val_epoch%ditr%d_c.jpg %(epoch,iteration)
```

### Testing

```
usage testing.py

1) Specify model weights in the python script testing.py.
1) The value of average Jaccard Index is printed on the terminal.
2) Output images are save under results folder as:
i)  INPUT_IMAGE: val_epoch%ditr%d_a.jpg %(epoch,iteration)
ii) LABEL: val_epoch%ditr%d_b.jpg %(epoch,iteration)
iii) OUTPUT_MAP: val_epoch%ditr%d_c.jpg %(epoch,iteration)
```

## Classification

The weights for the Classification model are available on: https://drive.google.com/file/d/12XZrDWHODCpfCB6Vh9TQdFKHP51ORcKp/view?usp=sharing

### Training

```
usage Train.py [--lr LEARNING_RATE] [--lr_de LEARNING_RATE_DECAY] [--checkpoint LOAD_PREV_CHECKPOINT] [-wd WEIGHT_DECAY] [-rd ROOT_DIR]

1) Checkpoints are saved at every epoch as checkpoint_ep%d.pth.tar %(epoch)
2) A list of loss and accuaracy values are saved as train_loss.pkl | val_loss.pkl and train_corrects.pkl | val_corrects.pkl

```

### Testing

```
usage Test.py

1) The scores for each images is saved in a numpy file as submission.npy
2) Run plot_roc.py to generate various plots.

```

## Acknowledgments

We thank Dr. Marius Bojarski NVIDIA Corporation for inpirations and useful feedbacks.



