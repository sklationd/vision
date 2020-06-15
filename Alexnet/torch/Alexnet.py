import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim 

import matplotlib
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import pyplot    
from matplotlib.pyplot import subplot     
from sklearn.metrics import accuracy_score

import json
import os

if(torch.cuda.is_available()):
    GPU = True
else:
    GPU = False
    print("It does not support CPU only execution for now")
    exit()

# For consistency of random split data
SEED = 145
torch.manual_seed(SEED)

# You should download ILSVRC2012 data manually, and locate it to ../../imagenet directory befor execute this cell
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

traindir = os.path.join('../../data/imagenet', 'train')
testdir   = os.path.join('../../data/imagenet',   'val')

input_dataset = datasets.ImageFolder(traindir, transform)
test_dataset = datasets.ImageFolder(testdir, transform)

train_size = int(0.9 * len(input_dataset))
val_size = len(input_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(input_dataset, [train_size, val_size])

batch   = 200
workers = 6

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=batch, num_workers=workers, pin_memory=False)
val_loader   = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=batch, num_workers=workers, pin_memory=False)
test_loader  = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch, num_workers=workers, pin_memory=False)

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((13, 13))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 13 * 13, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

net = AlexNet()

if(GPU):
    net.cuda()

params = list(net.parameters())

# set up loss function as Cross-Entropy Loss
loss_func = torch.nn.CrossEntropyLoss().cuda()
       
# SGD for optimization
optimization = torch.optim.Adam(net.parameters(),lr=0.00001, weight_decay=0.00001)

# load classes
idx2label = []
cls2label = {}
with open("../../data/imagenet/imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

numEpochs = 3    
validation_accuracy=[]

# if already done, pass learning
save_path = './Imagenet-Alexnet-'+str(numEpochs)+'.pth'
if os.path.isfile(save_path):    
    net.load_state_dict(torch.load(save_path))
    net.cuda()
    passplot = True
else:
    for epoch in range(numEpochs):

        path = './Imagenet-Alexnet-'+str(epoch+1)+'.pth'

        if os.path.isfile(path):
            net = AlexNet()
            net.load_state_dict(torch.load(path))
            net.cuda()
        else:
            epoch_training_loss = 0.0
            num_batches = 0
            
            for batch_num, training_batch in enumerate(train_loader, 0):
                # split training data into inputs and labels
                inputs, labels = training_batch
                
                # wrap data in 'Variable'
                inputs, labels = inputs.cuda(), labels.cuda()
                
                # Make gradients zero for parameters 'W', 'b'
                optimization.zero_grad()         
                
                # forward, backward pass with parameter update
                forward_output = net(inputs)
                loss = loss_func(forward_output, labels)

                loss.backward()   
                optimization.step()     

                # calculating loss 
                epoch_training_loss += loss.item()
                num_batches += 1
                
                if batch_num % 100 == 99:
                    t = epoch_training_loss / 100.0
                    epoch_training_loss = 0
                    print("epoch:", epoch, "|", "batch: ", batch_num, "|", "total:",batch*batch_num, "|", "loss:", t)

            torch.save(net.state_dict(), path)

        # calculate validation set accuracy
        accuracy = 0.0 
        num_batches = 0
        for batch_num, validation_batch in enumerate(val_loader):
            num_batches += 1
            inputs, actual_val = validation_batch

            # inferencing
            predicted_val = net(torch.autograd.Variable(inputs.cuda()))    
            predicted_val = predicted_val.cpu().data.numpy()
            predicted_val = np.argmax(predicted_val, axis = 1)

            # accuracy        
            accuracy += accuracy_score(actual_val.numpy(), predicted_val)
            
            if batch_num % 100 == 99:
                print("batch",batch_num,"accuracy:",accuracy/num_batches)

        print("----------------------------------------")
        print("epoch", epoch, "accuracy: ", accuracy/num_batches)

        validation_accuracy.append(accuracy/num_batches)

print('Finished Training')

epochs = list(range(numEpochs))

# plotting training and validation accuracies
if not passplot:
    fig1 = pyplot.figure()
    pyplot.plot(epochs, validation_accuracy, 'g', label="Validation")
    pyplot.xlabel("Epochs")
    pyplot.ylabel("Accuracy")
    pyplot.legend() 
    pyplot.savefig('../images/plot.png')
    pyplot.show(fig1)

class_correct = list(0. for i in range(1000))
class_total = list(0. for i in range(1000))

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        
        c = (predicted == labels).squeeze()
        for i in range(list(labels.shape)[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

accuracy_sum=0
count=0

for i in range(1000):
    if class_total[i]==0:
        continue
    count+=1
    temp = 100 * class_correct[i] / class_total[i]
    print('Accuracy of %30s : %2d %%' % (
        idx2label[i], temp))
    accuracy_sum+=temp
print('Accuracy average: ', accuracy_sum/count)