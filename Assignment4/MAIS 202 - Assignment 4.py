#!/usr/bin/env python
# coding: utf-8

# # Assignment 4: Mid Point Review

# Congratulations, you have reached the midpoint of the course! The rest of the course will be focused on more advanced state of the art machine learning techniques. However, before we dive into that, this assignment will be focused on ensuring you understand all the core concepts that have been covered so far. Keep in mind, these are questions that can be asked during machine learning internship interviews, so do make sure you understand them if you want to dive into this industry!

# ### 1) Linear vs Polynomial Regression
# - Describe both Linear Regression and Polynomial Regression (3 lines or less each).
# 
# - Describe overfitting vs underfitting with respect to parameters.  
# 

# In[12]:


## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
#Linear regression assumes the dataset has a linear distribution. Graphically,it tries to draw
#a line that represents the dataset points as much as possible. To predict a certain target,
#it places a point on the line with the other features.
#Polynomial works similarly to linear regression except that it is a more generalized version. It is still
#linear with respect to the parameters but the features can now have exponents. This allows for more general
#distributions
#Underfitting occurs when the model is not adapted enough to the training dataset, meaning the model has not
#been trained enough. Overfitting occurs when the model is too adapted to the training set, so much so that its
#performance is low on new vectors. This can occur when the degree of the polynomial regression is too large 
#for example.


# ### 2) Logistic Regression vs. Linear SVM
# - Describe how logistic regression works (3 lines or less)
# - Describe how linear SVM works. Mention the role(s) of:
#     - support vectors
#     - margin
#     - slack variables
#     - kernels
# - Plot an example for SVM where the linear kernel is not enough to separate the data, but another kernel works

# In[42]:


import matplotlib.pyplot as plt

from matplotlib.pyplot import imread
from sklearn.decomposition import PCA
import numpy as np

x = [0,30,60,90,120,150,180,210,240,270,300,330]
y = np.ones(len(x))
z = 1.5* np.ones(len(x))
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
c = ax.scatter(x, y, cmap='hsv', alpha=0.75)
c = ax.scatter(x, z, cmap='hsv', alpha=0.75)

## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
#Logistic regression is used to classify binary variables based on a set of parameters. It uses the sigmoid
#function to draw a curve that separates one class from the other. Everything that falls on one side of the
#curve belong to class A, everything on the other side to class B.
#Support Vector Machines identifies a hyper plane to discriminate observations. The coordinates of a single observation
#are called support vector. When multiple hyper planes are possible, SVMs will try to maximize the margin, i.e the
#distance between the hyper-plane and the observation points. If the observations cannot be separated by a hyper-plane,
#the kernel trick consists in mapping to higher dimensional spaces where such a separation is possible. In the below
#plotted example, we need to move to x^2 and y^2 for the svm to work


# ### 3) Linear SVM vs k-NN
# - K-Nearest Neighbours is a popular unsupervised learning algorithm. Explain the difference between supervised and unsupervised learning?
# - K-NN is an example of a lazy learning algorithm. Why is it called so. What could be a use case? Justify using a lazy learning algorithm in that case.
# - Outline the main steps for the KNN algorithm. Use text, code, plots, diagrams, etc as necessary.  
# - Plot a example dataset which works in an SVM classification and not k-NN classification. Repeat for the reverse scenario.

# In[45]:


## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
# Supervised learning is when there is a ground truth, so we know what the output values should be. In unsupervised
#learning, the algorithm tries to find relationships without there being a reference to compare to
#Lazy algorithms are those that don't require prior learning, all learning is done at the time of the prediction
#The major steps for the K-NN algorithm (also oulined in the comments below) are the following:
#1. Load the dataset
#2. Calculate the euclidian distance between the test point and all the points in the training dataset
#3. Sort the points by increasing order of euclidian distances
#4. Pick the first k elements of the above list, these are the k nearest neighbors
#5. Vote on the class, i.e choose the majority class found in the neighbors.

#In the above example, SVM would work better than KNN as blue and orange could be considered neighbors, while the SVM 
#would draw a circle and separate them. The reverse scenario would occur when the data is very noisy, the SVM
#would not be able to draw a hyperplane while KNN could adapt to the more complex classification


# ### 4) K-NN Implementation
# - Implement the K-NN algorithm by hand (ie. Don't use the sklearn implementation).

# In[55]:


# Implement kNN by hand. It might be useful to store all distances in one array/list

import pandas as pd
from sklearn.datasets import load_iris
import operator

def euclideanDistance(point1, point2, n):
    distanceSquare = 0
    for coordinate in range(n):
        distanceSquare += np.square(point2[coordinate] - point1[coordinate])
    return np.sqrt(distanceSquare)

# loading dataset
iris = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# Preview dataset
iris_df.head()
train, valid = np.split(iris_df, [int(0.7*len(iris_df))])
valid = np.array(valid)
train = np.array(train)

## YOUR CODE HERE
def takeDistance(elem):
    return elem[0]

def knn(trainingSet, testInstance, k):
 
    distances = []
    
    # Calculate and store Euclidian distance between each row and the test instance
    for x in range(trainingSet.shape[0]):
        distances.append((euclideanDistance(testInstance, trainingSet[x], len(testInstance)), trainingSet[x]))
 
    # Sort the distances array in increasing order
    sorted_distances = sorted(distances, key=takeDistance)
 
    neighbors = []
    
    # Extract k-nearest neighbors
    for x in range(k):
        neighbors.append(sorted_distances[x])
    #### End of STEP 3.3
    classVotes = {}
    
    #### Start of STEP 3.4
    # Calculating the most freq class in the neighbors
    for x in range(k):
        response = neighbors[x][1][-1]
 
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #### End of STEP 3.4

    #### Start of STEP 3.5
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0])
    #### End of STEP 3.5
    
print(knn(train, valid[0].reshape(-1), 5))


# ### 5) Ensemble Methods
# - Explain bagging and boosting. Clearly illustrate the difference between these methods. When would you use either one?
# - What is a decision tree? What is a random forest? Compare them and list 3 pros and cons of each?

# In[ ]:


## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
#A decision tree is a sequence of splits between features that allow us to reach a decision at the end. 
#Bagging refers to all algorithms that average multiple estimates to reduce variance. For example, in a random forest,
#every decision tree is made from a sample of the training drawn randomly. That way, in the forest, every tree acts
#as a different estimator and averaging from multiple estimators reduces variance.
#Boosting is used to reduce bias by using a sequence of weak learners, which, when combined, give a stronger learner.
#Individual decision trees risk to incur overfitting, but grouping them in random forests reduces that. This causes
#random forest to be slower though. Both can handle numerical and categorical data and are easy to read and interpret


# ### 6) PCA vs Autoencoders
# - Describe how PCA achieves dimensionality reduction. Outline the main steps of the algorithm
# - What is the importance of eigenvectors and eigenvalues in the PCA algorithm above.
# - When we compute the covariance matrix in PCA, we have to subtract the mean. Why do we do this?
# - What is Autoencoder (compare it to PCA)? Why are autoencoders better in general.
# - When is the reduced dimension of an encoder equivalent to that of a PCA

# In[58]:


## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
#Given the features matrix X, PCA achieves dimensionality reduction as follow
#1. The covariance matrix is computed by X^T * X
#2. The eigenvalues and eigenvectors of this matrix are computed
#3. Standardize by subtracting the mean from all the entries and then dividing by the standard deviation
#4. Sort the eigenvectors by the magnitude of their corresponding eigenvalues
#5. Multiply this matrix of eigenvectors by the standardized X
#6. In this last matrix, every column represents an observation dependent on all the initial variables, so one could
#could either choose a certain number of columns, or keep choosing columns until a certain variance threshold is reached

#The importance of the eigenvectors and eigenvalues is that they repsectively represent the direction and
#magintude of variance of the data.

#While PCA is limited to linear mapping, autoencoders can have nonlinear encoders and decorders. A single layer
#autoencoder with a linear transfer function can thus be interpreted as equivalent to PCA


# ### 7) Implementation
# 
# In the 1980's', Alex 'Sandy' Pentland came up with 'EigenFaces'. A novel way for facial classification using dimensionality reduction. We are going to try replicate the experiment in this question. We have loaded the face dataset for you below. Here's some steps for you: 
# 
# - Use PCA to reduce its dimensionality.
# - Use any algorithm to train a classifier for the dataset. You may use sklearn or pytorch. (Refer to PCA demo notebook for hints)
# - (Optional) Use autoencoders for the dimensionality reduction, compare results to PCA. Any comments/conculsions?
# 

# In[1]:


# loading the faces dataset
from sklearn.datasets import fetch_lfw_people


# uncomment below to load dataset(takes ~5 mins to load data)
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# assigning features vectors
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# In[20]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error

# plot an example image
plt.imshow(X[1].reshape(h,w), cmap = 'gray')

pca = PCA(n_components=200)
pca.fit(X)

X_pca = pca.transform(X)
print("Shape of data before PCA: {0}".format(X.shape))
print("Shape of data after PCA: {0}".format(X_pca.shape))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_pca, y)
y_pred = clf.predict(X_pca)
print('mse: ', mean_squared_error(y, y_pred))

### insert your code here ###


# ## Bonus Challenge! (Optional)
# 
# This will take some time. However, trust that it is a rewarding experience. There will be a prize for whoever implements it correctly!
# 
# - Implement a feed forward neural network with back proprogation using stochastic gradient descent by hand. 
# - Use any dataset you want and test the accuracy

# In[60]:


### your code below ###

