#!/usr/bin/env python
# coding: utf-8

# # Homework 3: Introduction to PyTorch

# PyTorch is a framework for creating and training neural networks. It's one of the most common neural network libraries, alongside TensorFlow, and is used extensively in both academia and industry. In this homework, we'll explore the basic operations within PyTorch, and we'll design a neural network to classify images.

# Let's start by importing the libraries that we'll need:

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# If you can't import torch, go to www.pytorch.org and follow the instructions there for downloading PyTorch. You can select CUDA Version as None, as we won't be working with any GPUs on this homework.

# ## PyTorch: Tensors

# In PyTorch, data is stored as multidimensional arrays, called tensors. Tensors are very similar to numpy's ndarrays, and they support many of the same operations. We can define tensors by explicity setting the values, using a python list:

# In[ ]:


A = torch.tensor([[1, 2], [4, -3]])
B = torch.tensor([[3, 1], [-2, 3]])

print("A:")
print(A)

print('\n')

print("B:")
print(B)


# Just like numpy, PyTorch supports operations like addition, multiplication, transposition, dot products, and concatenation of tensors. Look up and fill in the operations for the following:

# In[ ]:


print("Sum of A and B:")
### YOUR CODE HERE

print('\n')

print("Elementwise product of A and B:")
### YOUR CODE HERE

print('\n')

print("Matrix product of A and B:")
### YOUR CODE HERE

print('\n')

print("Transposition of A:")
### YOUR CODE HERE

print('\n')

print("Concatenation of A and B in the 0th dimension:")
### YOUR CODE HERE

print('\n')

print("Concatenation of A and B in the 1st dimension:")
### YOUR CODE HERE


# PyTorch also has tools for creating large tensors automatically, without explicity specifying the values. Find the corresponding tensor initialzers and fill in below. Your print statements should look like the following:
# 
# 3x4x5 Tensor of Zeros:
# ```
# tensor([[[0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.]],
# 
#         [[0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.]],
# 
#         [[0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.]]])
# ```
# 
# 
# 5x5 Tensor with random elements sampled from a standard normal distrubtion: (these should be randomly generated values)
# ```
# tensor([[ 0.2850,  0.5033, -1.8570, -1.6525,  0.3613],
#         [-0.7505,  0.4573, -0.2454,  0.1668,  0.7241],
#         [ 0.2976,  0.9827, -0.4879, -1.1144, -1.8235],
#         [-0.0264,  0.7341, -0.2235,  0.5306,  0.8385],
#         [ 0.2740,  0.3522, -0.5244, -0.1132,  0.5135]])
# ```
# 
# Tensor created from a range:
# ```
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# ```
# 

# In[ ]:


print("3x4x5 Tensor of Zeros:")
### YOUR CODE HERE

print('\n')

print("5x5 Tensor with random elements sampled from a standard normal distrubtion:")
### YOUR CODE HERE

print('\n')

print("Tensor created from a range:")
### YOUR CODE HERE


# Now, use PyTorch tensors to complete the following computation:
# 
# Create a tensor of integers from the range 0 to 99, inclusive. Add 0.5 to each element in the tensor, and square each element of the result. Then, negate each element of the tensor, and apply the exponential to each element (i.e., change each element x into e^x). Now, sum all the elements of the tensor. Multiply this tensor by 2 and square each element and print your result.
# 
# If you're right, you should get something very close to $$\pi \approx 3.14 .$$

# In[ ]:


val = torch.arange(100).float()

### YOUR CODE HERE

print(val)


# Now we'll try writing a computation that's prevalent throughout a lot of deep learning algorithms - calculating the softmax function:
# $$softmax(x_i) = \frac{e^{x_i}}{\sum_{j = 0}^{n - 1} e^{x_j}}$$
# Calculate the softmax function for the $val$ tensor below where $n$ is the number of elements in $val$, and $x_i$ is each element in $val$. DO NOT use the built-in softmax function. We should end up with a tensor that represents a probability distribution that sums to 1. (hint: you should calculate the sum of the exponents first)

# In[ ]:


val1 = torch.arange(10).float()

### YOUR CODE HERE

print(result1)
print(torch.sum(result1))


# To do this, you'll need to use the PyTorch documentation at https://pytorch.org/docs/stable/torch.html. Luckily, PyTorch has very well-written docs.

# ## PyTorch: Autograd

# Autograd is PyTorch's automatic differentiation tool: It allows us to compute gradients by keeping track of all the operations that have happened to a tensor. In the context of neural networks, we'll interpret these gradient calculations as backpropagating a loss through a network.

# To understand how autograd works, we first need to understand the idea of a __computation graph__. A computation graph is a directed, acyclic graph (DAG) that contains a blueprint of a sequence of operations. For a neural network, these computations consist of matrix multiplications, bias additions, ReLUs, softmaxes, etc. Nodes in this graph consist of the operations themselves, while the edges represent tensors that flow forward along this graph.

# In PyTorch, the creation of this graph is __dynamic__. This means that tensors themselves keep track of their own computational history, and this history is build as the tensors flow through the network; this is unlike TensorFlow, where an external controller keeps track of the entire computation graph. This dynamic creation of the computation graph allows for lots of cool control-flows that are not possible (or at least very difficult) in TensorFlow.

# ![alt text](https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/dynamic_graph.gif)
# <center>_Dynamic computation graphs are cool!_</center>
# _ _

# Let's take a look at a simple computation to see what autograd is doing. First, let's create two tensors and add them together. To signal to PyTorch that we want to build a computation graph, we must set the flag requires_grad to be True when creating a tensor.

# In[ ]:


a = torch.tensor([1, 2], dtype=torch.float, requires_grad=True)
b = torch.tensor([8, 3], dtype=torch.float, requires_grad=True)

c = a + b


# Now, since a and b are both part of our computation graph, c will automatically be added:

# In[ ]:


c.requires_grad


# When we add a tensor to our computation graph in this way, our tensor now has a grad_fn attribute. This attribute tells autograd how this tensor was generated, and what tensor(s) this particular node was created from.

# In the case of c, its grad_fn is of type AddBackward1, PyTorch's notation for a tensor that was created by adding two tensors together:

# In[ ]:


c.grad_fn


# Every grad_fn has an attribute called next_functions: This attribute lets the grad_fn pass on its gradient to the tensors that were used to compute it.

# In[ ]:


c.grad_fn.next_functions


# If we extract the tensor values corresponding to each of these functions, we can see a and b! 

# In[ ]:


print(c.grad_fn.next_functions[0][0].variable)
print(c.grad_fn.next_functions[1][0].variable)


# In this way, autograd allows a tensor to record its entire computational history, implicitly creating a computational graph -- All dynamically and on-the-fly!

# ## PyTorch: Modules and Parameters

# In PyTorch, collections of operations are encapsulated as __modules__. One way to visualize a module is to take a section of a computational graph and collapse it into a single node. Not only are modules useful for encapsulation, they have the ability to keep track of tensors that are contained inside of them: To do this, simply wrap a tensor with the class torch.nn.Parameter.

# To define a module, we must subclass the type torch.nn.Module. In addition, we must define a _forward_ method that tells PyTorch how to traverse through a module.

# For example, let's define a logistic regression module. This module will contain two parameters: The weight vector and the bias. Calling the _forward_ method will output a probability between zero and one.

# In[ ]:


class LogisticRegression(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10))
        self.bias = nn.Parameter(torch.randn(1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, vector):
        return self.sigmoid(torch.dot(vector, self.weight) + self.bias)
        


# Note that we have fixed the dimension of our weight to be 10, so our module will only accept 10-dimensional data.

# We can now create a random vector and pass it through the module:

# In[ ]:


module = LogisticRegression()
vector = torch.randn(10)
output = module(vector)


# In[ ]:


output


# Now, say that our loss function is mean-squared-error and our target value is 1. We can then write our loss as:

# In[ ]:


loss = (output - 1) ** 2


# In[ ]:


loss


# To minimize this loss, we just call loss.backward(), and all the gradients will be computed for us! Note that wrapping a tensor as a Parameter will automatically set requires_grad = True.

# In[ ]:


loss.backward()


# In[ ]:


print(module.weight.grad)
print(module.bias.grad)


# ## Fully-connected Networks for Image Classification

# Using this knowledge, you will create a neural network in PyTorch for image classification on the CIFAR-10 dataset. PyTorch uses the $DataLoader$ class for you to load data into batches to feed to your learning algorithms - we highly suggest you familiarze yourself with this as well as the Dataset API here: https://pytorch.org/docs/stable/data.html. Fill in the below code to instantiate 3 DataLoaders for your training, validation and test sets. We would prefer that you NOT use the `torchvision.transform` API - we want you to get some practice in data preprocessing! Here are the transformations we want you to perform:
# 1. Split the `val_and_test_set` into two separate datasets (each with 5000 elements)
# 2. Convert all the `np.array` elements into `torch.tensor` elements.
# 3. All values will be pixel values in our images are in the range of [0, 256]. Normalize this so that each pixel is in the range [0, 1].
# 3. Flatten all images. All your images will be of shape (32, 32, 3), we need them as flat (32 * 32 * 3) size tensors as input to our neural network.
# 4. Load everything into a DataLoader. (check how this works in the PyTorch docs!) 
# 
# Be sure to have the options `shuffle=True` (so that your dataset is shuffled so that samples from the dataset are not correlated) and also `batch_size=32` or larger. This is a standard minibatch size. If you're curious about what batch size does (and are somewhat familiar with statistics), here's a great answer https://stats.stackexchange.com/questions/316464/how-does-batch-size-affect-convergence-of-sgd-and-why.

# In[2]:


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)
#trainset = [(np.asarray(image) / 256, label) for image, label in trainset]
Trsampler = torch.utils.data.SubsetRandomSampler(np.arange(5000,25000, dtype=np.int64))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4096, sampler = Trsampler, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)

Vsampler = torch.utils.data.SubsetRandomSampler(np.arange(0, 5000, dtype=np.int64))
Tsampler = torch.utils.data.SubsetRandomSampler(np.arange(0, 5000, dtype=np.int64))

Valid_loader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          sampler = Vsampler, shuffle=False, num_workers=2)
Test_loader = torch.utils.data.DataLoader(testset, batch_size=256,
                                          sampler = Tsampler, shuffle=False, num_workers=2)
#val_and_test_set = [(np.asarray(image) / 256, label) for image, label in val_and_test_set]



classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# CIFAR-10 consists of 32 x 32 color images, each corresponding to a unique class indicating the object present within the image. Use Matplotlib to print out the first few images.

# In[ ]:


for images, x in trainloader:
    print(images.shape[0])


# Now try to build and train a plain neural network that properly classifies images in the CIFAR-10 dataset. Try to achieve at least around 40% accuracy (the higher the better!).
# 
# Take a look at the PyTorch documentation for some help in how to do this.
# 
# Google is your friend -- Looking things up on the PyTorch docs and on StackOverflow will be helpful.

# In[3]:


class NeuralNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super().__init__()
        self.weight_1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.weight_2 = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.bias_1 = nn.Parameter(torch.full((1, hidden_dim), 1.41414141))
        self.bias_2 = nn.Parameter(torch.full((1, output_dim), 1.41414141))
        self.sigmoid = nn.Sigmoid()
        ### YOUR CODE HERE
        
    def forward(self, data):
        
        ### YOUR CODE HERE
        data = data.view(data.size()[0], -1)
        
        hidden = self.sigmoid(torch.mm(data, self.weight_1) + self.bias_1)
        return self.sigmoid(torch.mm(hidden, self.weight_2) + self.bias_2)
        


# The following NN works but is of only 11% accuracy :'(

# In[ ]:


class NeuralNet2(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        data = nn.functional.relu(self.conv1(data))
        data = self.pool(data)
        data = data.view(-1, 24*16*16)
        data = torch.nn.functional.relu(self.fc1(data))
        data = self.fc2(data)
        return(data)


# In[ ]:


def reshape(images):
    '''
    Reshapes a set of images of the shape (batch_size, width, height, channels)
    into the proper shape (batch_size, width * height * channels) that the model can accept.
    '''
    x = images.reshape(images.shape[0], -1).float
    return x


# The following cell was for initial testing purpose which only gave me an accuracy of 11% :'(

# In[ ]:


from tqdm import tqdm
from torch.autograd import Variable
EPOCHS = 2
LEARNING_RATE = 0.01
INPUT_SIZE = 24*16*16
HIDDEN_SIZE = 640

OUTPUT_SIZE = 10

net = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
secondnet = NeuralNet2(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
### Define an optimizer and a loss function here. We pass our network parameters to our optimizer here so we know
### which values to update by how much.
optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

for epoch in tqdm(range(EPOCHS)):
    
    total_loss = 0
    
    for images, labels in trainloader:
        print(images.size())
        #data = reshape(images)
        
        ### <YOUR CODE HERE> ####
        outputs = secondnet(Variable(images))
        loss = loss_fn(outputs, Variable(labels))
        
        # Zero gradients, call .backward(), and step the optimizer.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ### </YOUR CODE HERE> ###
        
        total_loss += loss.item()
        
    average_loss = total_loss / len(trainloader)
    
    val_loss = 0
    for inputs, labels in Valid_loader:
            
        inputs, labels = Variable(inputs), Variable(labels)

        val_outputs = secondnet(inputs)
        val_loss_size = loss_fn(val_outputs, labels)
        val_loss += val_loss_size.item()
    true_val_loss = val_loss/len(Valid_loader)
    
    print("(epoch, train_loss, val_loss) = ({0}, {1}, {2})".format(epoch, average_loss, true_val_loss))


# In[7]:


from tqdm import tqdm
from torch.autograd import Variable
EPOCHS = 64
LEARNING_RATE = 0.01
INPUT_SIZE = 3*32*32
HIDDEN_SIZE = 640

OUTPUT_SIZE = 10

net = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
### Define an optimizer and a loss function here. We pass our network parameters to our optimizer here so we know
### which values to update by how much.
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

for epoch in tqdm(range(EPOCHS)):
    
    total_loss = 0
    
    for images, labels in trainloader:
        #data = reshape(images)
        data = torch.as_tensor(images)
        ### <YOUR CODE HERE> ####
        outputs = net(data)
        loss = loss_fn(outputs, Variable(labels))
        
        # Zero gradients, call .backward(), and step the optimizer.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ### </YOUR CODE HERE> ###
        
        total_loss += loss.item()
        
    average_loss = total_loss / len(trainloader)
    
    val_loss = 0
    for inputs, labels in Valid_loader:
            
        inputs, labels = Variable(inputs), Variable(labels)

        val_outputs = net(inputs)
        val_loss_size = loss_fn(val_outputs, labels)
        val_loss += val_loss_size.item()
        
    true_val_loss = val_loss/len(Valid_loader)
    print("(epoch, train_loss, val_loss) = ({0}, {1}, {2})".format(epoch, average_loss, true_val_loss))


# In[8]:


### YOUR CODE HERE - Here, we test the overall accuracy of our model.
    
correct = 0
total = 0
with torch.no_grad():
    for data in Test_loader:
        images, labels = data
        images, labels = Variable(images), Variable(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 5000 test images: %d %%' % (
    100 * correct / total))


# ## Submission
# For submiting, please download this notebook as a `.py` file. To do so, click on `File -> Download as -> Python (.py)`. Put the downloaded `assignment3.py` into this folder and commit the file.

# ## Additional Resources
# If you're interested in using PyTorch as a framework for deep learning (especially for your final projects! We highly recommend you use this!), check out the PyTorch tutorials: https://pytorch.org/tutorials/. They have tutorials for everything from image to text to reinforcement learning tasks.
