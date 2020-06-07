
# coding: utf-8

# ### Implementing LeNet-5 Architecture On MNIST Dataset (GPU Implementation)

# In[1]:

import torch
torch.multiprocessing.set_start_method("spawn")
import torch.nn   
import torch.optim 
import torch.nn.functional 
import torchvision.datasets   
import torchvision.transforms     

import numpy as np
import matplotlib
matplotlib.use('Agg')       
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot    
from matplotlib.pyplot import subplot     
from sklearn.metrics import accuracy_score

if(torch.cuda.is_available()):
    GPU = True
else:
    GPU = False
    print("It does not support CPU only execution for now")
    exit()

print("******* IMPORTING *******")
# In[2]:

transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                               torchvision.transforms.Normalize([0.5],[0.5])])
train = torchvision.datasets.MNIST(root='../../data', train=True, download=True, transform=transformImg)
valid = torchvision.datasets.MNIST(root='../../data', train=True, download=True, transform=transformImg)
test = torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transformImg)  

# Seperate training and validation set (8 : 2)
idx = list(range(len(train)))
np.random.seed(1009)
np.random.shuffle(idx)          
train_idx = idx[ : int(0.8 * len(idx))]       
valid_idx = idx[int(0.8 * len(idx)) : ]

print("******* LOADING DATA *******")
# In[3]:

# sample images
fig1 = train.train_data[0].numpy()  
fig2 = train.train_data[1000].numpy()
fig3 = train.train_data[20000].numpy()  
fig4 = train.train_data[50000].numpy()
subplot(2,2,1), pyplot.imshow(fig1)  
subplot(2,2,2), pyplot.imshow(fig2) 
subplot(2,2,3), pyplot.imshow(fig3)
subplot(2,2,4), pyplot.imshow(fig4)

print("******* PREVIEW DATA *******")
# In[4]:

# generate training and validation set samples
train_set = torch.utils.data.sampler.SubsetRandomSampler(train_idx)    
valid_set = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)  

train_loader = torch.utils.data.DataLoader(train, batch_size=30, sampler=train_set, num_workers=0)  
valid_loader = torch.utils.data.DataLoader(train, batch_size=30, sampler=valid_set, num_workers=0)    
test_loader = torch.utils.data.DataLoader(test, num_workers=0)       

print("******* SEPERATE TRAINING/VALIDATION DATA *******")
# In[5]:

# Defining the network (LeNet-5)  
class LeNet5(torch.nn.Module):          
     
    def __init__(self):     
        super(LeNet5, self).__init__()

        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        
        # Convolution
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2) 
        
        # Fully connected layer
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10) 
        
    def forward(self, x):

        # Convolution and re-sampling
        x = torch.nn.functional.relu(self.conv1(x))  
        x = self.max_pool_1(x) 
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)

        # Resizing
        x = x.view(-1, 16*5*5)

        # FC-layer (16*5*5 -> 120 -> 84 -> 10)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
     
net = LeNet5()     

if(GPU):
    net.cuda()

print("******* DEFINE NETWORK *******")
# In[6]:

# set up loss function as Cross-Entropy Loss
loss_func = torch.nn.CrossEntropyLoss()
       
# SGD for optimization
optimization = torch.optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)


# In[7]:

numEpochs = 20    
training_accuracy = []     
validation_accuracy = []

for epoch in range(numEpochs):
    
    epoch_training_loss = 0.0
    num_batches = 0
    for batch_num, training_batch in enumerate(train_loader):
        # split training data into inputs and labels
        inputs, labels = training_batch
        
        # wrap data in 'Variable'
        inputs, labels = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(labels.cuda())        
        
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
        
    print("epoch: ", epoch, ", loss: ", epoch_training_loss/num_batches)            
     
    # calculate training set accuracy
    accuracy = 0.0 
    num_batches = 0
    for batch_num, training_batch in enumerate(train_loader):
        num_batches += 1
        inputs, actual_val = training_batch

        # inferencing
        predicted_val = net(torch.autograd.Variable(inputs.cuda()))
        predicted_val = predicted_val.cpu().data.numpy()    # convert cuda() type to cpu(), then convert it to numpy
        predicted_val = np.argmax(predicted_val, axis = 1)  # retrieved max_values along every row    
        
        # accuracy   
        accuracy += accuracy_score(actual_val.numpy(), predicted_val)
    training_accuracy.append(accuracy/num_batches)   

    # calculate validation set accuracy 
    accuracy = 0.0 
    num_batches = 0
    for batch_num, validation_batch in enumerate(valid_loader):
        num_batches += 1
        inputs, actual_val = validation_batch

        # inferencing
        predicted_val = net(torch.autograd.Variable(inputs.cuda()))    
        predicted_val = predicted_val.cpu().data.numpy()
        predicted_val = np.argmax(predicted_val, axis = 1)

        # accuracy        
        accuracy += accuracy_score(actual_val.numpy(), predicted_val)
    validation_accuracy.append(accuracy/num_batches)


# In[8]:

epochs = list(range(numEpochs))

# plotting training and validation accuracies
fig1 = pyplot.figure()
pyplot.plot(epochs, training_accuracy, 'r', label="Training")
pyplot.plot(epochs, validation_accuracy, 'g', label="Validation")
pyplot.xlabel("Epochs")
pyplot.ylabel("Accuracy")
pyplot.legend() 
pyplot.savefig('../images/plot.png')
pyplot.show(fig1)


# In[9]:

# Validation with test dataset
correct = 0
total = 0
for test_data in test_loader:
    total += 1
    inputs, actual_val = test_data 

    # inferencing
    predicted_val = net(torch.autograd.Variable(inputs.cuda()))   
    
    # convert 'predicted_val' GPU tensor to CPU tensor and extract the column with max_score
    predicted_val = predicted_val.cpu().data
    max_score, idx = torch.max(predicted_val, 1)
    
    # compare it with actual value and estimate accuracy
    correct += (idx == actual_val).sum()
       
print("Classifier Accuracy: ", ((correct*100.0)/total).item())


# In[10]
PATH = './mnist_LeNet-5.pth'
torch.save(net.state_dict(), PATH)

# %%
