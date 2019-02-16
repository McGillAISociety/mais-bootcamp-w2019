#!/usr/bin/env python
# coding: utf-8

# # Homework 3: Introduction to PyTorch

# PyTorch is a framework for creating and training neural networks. It's one of the most common neural network libraries, alongside TensorFlow, and is used extensively in both academia and industry. In this homework, we'll explore the basic operations within PyTorch, and we'll design a neural network to classify images.

# Let's start by importing the libraries that we'll need:

# In[1]:


import torch
import torchvision
import torch.nn as nn

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# If you can't import torch, go to www.pytorch.org and follow the instructions there for downloading PyTorch. You can select CUDA Version as None, as we won't be working with any GPUs on this homework.

# ## PyTorch: Tensors

# In PyTorch, data is stored as multidimensional arrays, called tensors. Tensors are very similar to numpy's ndarrays, and they support many of the same operations. We can define tensors by explicity setting the values, using a python list:

# In[2]:


A = torch.tensor([[1, 2], [4, -3]])
B = torch.tensor([[3, 1], [-2, 3]])

print("A:")
print(A)

print('\n')

print("B:")
print(B)


# Just like numpy, PyTorch supports operations like addition, multiplication, transposition, dot products, and concatenation of tensors.

# In[3]:


print("Sum of A and B:")
print(torch.add(A, B))

print('\n')

print("Elementwise product of A and B:")
print(torch.mul(A, B))

print('\n')

print("Matrix product of A and B:")
print(torch.matmul(A, B))

print('\n')

print("Transposition of A:")
print(torch.t(A))

print('\n')

print("Concatenation of A and B in the 0th dimension:")
print(torch.cat((A, B), dim=0))

print('\n')

print("Concatenation of A and B in the 1st dimension:")
print(torch.cat((A, B), dim=1))


# PyTorch also has tools for creating large tensors automatically, without explicity specifying the values:

# In[4]:


print("3x4x5 Tensor of Zeros:")
print(torch.zeros(3, 4, 5))

print('\n')

print("5x5 Tensor with random elements sampled from a standard normal distrubtion:")
print(torch.randn(5, 5))

print('\n')

print("Tensor created from a range:")
print(torch.arange(10))


# Now, use PyTorch tensors to complete the following computation:
# 
# Create a tensor of integers from the range 0 to 99, inclusive. Add 0.5 to each element in the tensor, and square each element of the result. Then, negate each element of the tensor, and apply the exponential to each element (i.e., change each element x into e^x). Now, sum all the elements of the tensor. Multiply this tensor by 2 and square each element and print your result.
# 
# If you're right, you should get something very close to $$\pi \approx 3.14 .$$

# In[19]:


val = torch.arange(100).float()

### <YOUR CODE HERE> ####
#add 0.5 to each element
val += 0.5

#square each element
val = torch.mul(val, val)

#negate each element
val = torch.neg(val)

#apply exponential
val = torch.exp(val)

#sum all elements
val = torch.sum(val)

#multiply tensor by 2
val *=2

#square each element again
val = torch.mul(val, val)

### </YOUR CODE HERE> ###
print(sum_val)


# Now we'll try writing a computation that's prevalent throughout a lot of deep learning algorithms - calculating the softmax function:
# $$softmax(x_i) = \frac{e^{x_i}}{\sum_{j = 0}^{n - 1} e^{x_j}}$$
# Calculate the softmax function for the $val$ tensor below where $n$ is the number of elements in $val$, and $x_i$ is each element in $val$. DO NOT use the built-in softmax function. We should end up with a tensor that represents a probability distribution that sums to 1. (hint: you should calculate the sum of the exponents first)

# In[31]:


val1 = torch.arange(10).float()

### <YOUR CODE HERE> ####

#define a function to calculate softmax
def softmax(val):
    #get exponential of each element of val
    exp_val = torch.exp(val)
    #get sum of exponentials
    exp_sum =torch.sum(exp_val)
    
    #divide each element by sum of exponentials, return result
    return exp_val/exp_sum
    
#get result
result1 = softmax(val1)

### </YOUR CODE HERE> ###

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

# In[32]:


a = torch.tensor([1, 2], dtype=torch.float, requires_grad=True)
b = torch.tensor([8, 3], dtype=torch.float, requires_grad=True)

c = a + b


# Now, since a and b are both part of our computation graph, c will automatically be added:

# In[33]:


c.requires_grad


# When we add a tensor to our computation graph in this way, our tensor now has a grad_fn attribute. This attribute tells autograd how this tensor was generated, and what tensor(s) this particular node was created from.

# In the case of c, its grad_fn is of type AddBackward1, PyTorch's notation for a tensor that was created by adding two tensors together:

# In[34]:


c.grad_fn


# Every grad_fn has an attribute called next_functions: This attribute lets the grad_fn pass on its gradient to the tensors that were used to compute it.

# In[35]:


c.grad_fn.next_functions


# If we extract the tensor values corresponding to each of these functions, we can see a and b! 

# In[36]:


print(c.grad_fn.next_functions[0][0].variable)
print(c.grad_fn.next_functions[1][0].variable)


# In this way, autograd allows a tensor to record its entire computational history, implicitly creating a computational graph -- All dynamically and on-the-fly!

# ## PyTorch: Modules and Parameters

# In PyTorch, collections of operations are encapsulated as __modules__. One way to visualize a module is to take a section of a computational graph and collapse it into a single node. Not only are modules useful for encapsulation, they have the ability to keep track of tensors that are contained inside of them: To do this, simply wrap a tensor with the class torch.nn.Parameter.

# To define a module, we must subclass the type torch.nn.Module. In addition, we must define a _forward_ method that tells PyTorch how to traverse through a module.

# For example, let's define a logistic regression module. This module will contain two parameters: The weight vector and the bias. Calling the _forward_ method will output a probability between zero and one.

# In[37]:


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

# In[38]:


module = LogisticRegression()
vector = torch.randn(10)
output = module(vector)


# In[39]:


output


# Now, say that our loss function is mean-squared-error and our target value is 1. We can then write our loss as:

# In[40]:


loss = (output - 1) ** 2


# In[41]:


loss


# To minimize this loss, we just call loss.backward(), and all the gradients will be computed for us! Note that wrapping a tensor as a Parameter will automatically set requires_grad = True.

# In[42]:


loss.backward()


# In[43]:


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


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

val_and_test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

#images.reshape(images.shape[0], -1).float()

### YOUR CODE HERE
val_and_test_set = [(np.asarray(image) / 256, label) for image, label in val_and_test_set]

#trainloader
trainset = [(np.asarray(image) / 256, label) for image, label in trainset]
trainset = [(torch.tensor(image.reshape(1, -1)), torch.tensor(label)) for image, label in trainset]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

#valloader
valset = val_and_test_set[:5000]
valset = [(torch.tensor(image.reshape(1,-1)), torch.tensor(label)) for image, label in valset]
valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=True)

#test loader
testset = val_and_test_set[5000:]
testset = [(torch.tensor(image.reshape(1,-1)), torch.tensor(label)) for image, label in valset]
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# CIFAR-10 consists of 32 x 32 color images, each corresponding to a unique class indicating the object present within the image. Here are a few examples:

# In[59]:


### YOUR CODE HERE - Grab a few examples from trainset and plot them
for image, label in trainset[:4]:
    plt.title(classes[label])
    plt.imshow(image.reshape(32,32,-1))
    plt.show()
    
    #colours differ because we normalized them


# **Your assignment is to create and train a neural network that properly classifies images in the CIFAR-10 dataset. Try to achieve at least around 40% accuracy (the higher the better!).**
# 
# We've given you some starter code to achieve this task, but the rest is up to you. Google is your friend -- Looking things up on the PyTorch docs and on StackOverflow will be helpful.

# In[3]:


class NeuralNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super().__init__()
        
        ### <YOUR CODE HERE> ####
        #hidden layer
        #weights: each row corresponds to a different feature, each column corresponds to a node in the hidden layer
        self.weights1 = nn.Parameter(torch.randn(input_dim, hidden_dim).double())
        #bias:
        self.bias1 = nn.Parameter(torch.randn(1, hidden_dim).double())
        
        #output layer
        #weights: each row corresponds to a different output from the hidden layer, each column corresponds to a node in the output layer
        self.weights2 = nn.Parameter(torch.randn(hidden_dim, output_dim).double())
        #bias:
        self.bias2 = nn.Parameter(torch.randn(1, output_dim).double())
        
        #define activation function
        self.activation = nn.Sigmoid()
        
        ### </YOUR CODE HERE> ###
        
    def forward(self, data):
        
        ### <YOUR CODE HERE> ####
        #---------------hidden layer---------------
        #multiply inputs with weights of transfer function
        #each row of self.temp represents a different data entry (a different image)
        #each column of self.temp represents the transfer function output of a different node
        temp = torch.matmul(data, self.weights1) + self.bias1
        
        #activation function
        temp2 = self.activation(temp) 
        
        #---------------output layer---------------
        #transfer function
        temp3 = torch.matmul(temp2, self.weights2) + self.bias2
        
        #activation function
        output = self.activation(temp3) # final activation function
        
        return output.squeeze()
    
        ### </YOUR CODE HERE> ###
        


# In[7]:


EPOCHS = 5
LEARNING_RATE = 0.001
INPUT_SIZE = 32*32*3 #size corresponds with number of features
HIDDEN_SIZE = 1000

OUTPUT_SIZE = 10

net = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

### Define an optimizer and a loss function here. We pass our network parameters to our optimizer here so we know
### which values to update by how much.
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

print(net)

for epoch in range(EPOCHS):
    
    total_loss = 0
    
    for images, labels in trainloader:
        
        ### <YOUR CODE HERE> ####
        
        output = net(images)
        loss = loss_fn(output, labels)
        
        # Zero gradients, call .backward(), and step the optimizer.
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ### </YOUR CODE HERE> ###
        
        total_loss += loss.item()
        
    average_loss = total_loss / len(trainloader)
    
    ### Calculate validation accuracy here by iterating through the validation set.
    ### We use torch.no_grad() here because we don't want to accumulate gradients in our function.
    with torch.no_grad():
        val_acc = 0
        
        for images, labels in valloader:
            val_pred = net(images)
            val_argmax = torch.argmax(val_pred, dim=1)
            val_acc += torch.sum(val_argmax == labels).item()
    
    print("(epoch, train_loss, val_acc) = ({0}, {1}, {2})".format(epoch, average_loss, val_acc/float(len(valloader.dataset))))


# In[8]:


### Here, we test the overall accuracy of our model. ###
with torch.no_grad():
    test_acc = 0
        
    for images, labels in valloader:
        test_pred = net(images)
        test_argmax = torch.argmax(test_pred, dim=1)
        test_acc += torch.sum(test_argmax == labels).item()
        
    print("Test accuracy:", (test_acc/ float(len(testloader.dataset))))
    
    #we get a test accuracy of 35.88%


# ## Submission
# For submiting, please download this notebook as a `.py` file. To do so, click on `File -> Download as -> Python (.py)`. Put the downloaded `assignment3.py` into this folder and commit the file.
