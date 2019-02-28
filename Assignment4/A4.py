#!/usr/bin/env python
# coding: utf-8

# # Assignment 4: Mid Point Review

# Congratulations, you have reached the midpoint of the course! The rest of the course will be focused on more advanced state of the art machine learning techniques. However, before we dive into that, this assignment will be focused on ensuring you understand all the core concepts that have been covered so far. Keep in mind, these are questions that can be asked during machine learning internship interviews, so do make sure you understand them if you want to dive into this industry!

# ### 1) Linear vs Polynomial Regression
# - Describe both Linear Regression and Polynomial Regression (3 lines or less each).
# 
# - Describe overfitting vs underfitting with respect to parameters.  
# 

# ## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
# 
# 1. Linear regression: A linear approach to model the relationship between one (simple linear regression) or more (multiple linear regression) independent variables and a dependent variable. 
# 
# 2. Polynomial regression: The dependent variable is modelled as a polynomial of the independent variable. This model could then fit a nonlinear relationship but is still considered linear if  we consider the powers as new features. 
# 
# 3. underfitting: A low dimensional polynomial is not powerful enough to fit more complex models
# 
# 4. overfitting: Using a polynomial with unnecessarily high dimensions to fit model; that is, the model learn from the noise of the training set and therefore fails to generalize.
# 

# ### 2) Logistic Regression vs. Linear SVM
# - Describe how logistic regression works (3 lines or less)
# - Describe how linear SVM works. Mention the role(s) of:
#     - support vectors
#     - margin
#     - slack variables
#     - kernels
# - Plot an example for SVM where the linear kernel is not enough to separate the data, but another kernel works

# ## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
# 
# 1. Logistic regression: It is used particularly when the dependent variable is binary. 
#    y = e^(b0 + b1*x) / (1 + e^(b0 + b1*x))
#    
# 2. Firstly, we plot each data item as a point in n-dimensional space (where n is number of features) with the value of each feature being the value of a particular coordinate (support vectors). Then, we perform classification by finding the hyper-plane that differentiate the two classes very well; that is, 
#     (i) Select the hyper-plane which segregates the two classes better
#     (ii) Select the hyper-plane with higher margin
#     (iii) SVM selects the hyper-plane which classifies the classes accurately prior maximizing margin
#     (iv) SVM ignore outliers and find the hyper-plane that has maximum margin
#     (v) non separable case: kernels. These are functions which transform low dimensional input space to a higher dimensional space, and thus convert not separable problem to separable problem. Slack variables are used to allow acceptable errors to avoid overfitting.
#     
# 3.https://en.wikipedia.org/wiki/Support-vector_machine#/media/File:Kernel_Machine.svg

# ### 3) Linear SVM vs k-NN
# - K-Nearest Neighbours is a popular unsupervised learning algorithm. Explain the difference between supervised and unsupervised learning?
# - K-NN is an example of a lazy learning algorithm. Why is it called so. What could be a use case? Justify using a lazy learning algorithm in that case.
# - Outline the main steps for the KNN algorithm. Use text, code, plots, diagrams, etc as necessary.  
# - Plot a example dataset which works in an SVM classification and not k-NN classification. Repeat for the reverse scenario.

# ## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
# 
# 1. Supervised learning is where you have input variables and an output variable, and you use an algorithm to learn the mapping function from the input to the output. Supervised learning problems can be grouped into regression and classification.
#     Unsupervised learning is where you only have input data and NO corresponding output variables. The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data.
# 
# 2. K-NN is a lazy learner because it doesn’t learn a discriminative function from the training data but “memorizes” the training dataset, and therefore it does't have training phase. KNN Algorithm is based on feature similarity: How closely out-of-sample features resemble our training set determines how we classify a given data point. e.g majority vote classification
# 
# 3.  (i) A positive integer k is specified, along with a new sample
#     (ii) We select the k entries in our database which are closest to the new sample
#     (iii) We find the most common classification of these entries
#     (iv) This is the classification we give to the new sample
# 

# ### 4) K-NN Implementation
# - Implement the K-NN algorithm by hand (ie. Don't use the sklearn implementation).

# In[1]:


# Implement kNN by hand. It might be useful to store all distances in one array/list

import pandas as pd
import numpy as np
import operator
from sklearn.datasets import load_iris

# loading dataset
iris = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# Preview dataset
#print(iris_df.head())

## YOUR CODE HERE

df = iris_df.sample(frac=1).reset_index(drop=True)
train=df.iloc[:100]
test=df.iloc[100:]
trainx = train.drop(columns = "target")
trainy = train.target
testx=test.drop(columns="target")
testy=test.target

# step 1: 
import math
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

# step 2:
def knn(trainingSet, testInstance, k):
    distances = {}
    sort = {}
    length = testInstance.shape[0]
    
# step 3:
    for x in range(len(trainingSet)): #range 0-99
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length) #return a distance
        #print(dist)
        distances[x] = dist 
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    classVotes = {}
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# ...

def accuracy(target, predict):
    return (sum(target == predict) / float(target.size))*100

predictions=[]
k = 3
for x in range(len(testx)):
    result = knn(train,testx.iloc[x], k)
    predictions.append(result)

accuracy = accuracy(testy,predictions)
print('Accuracy: ' + str(accuracy))


# ### 5) Ensemble Methods
# - Explain bagging and boosting. Clearly illustrate the difference between these methods. When would you use either one?
# - What is a decision tree? What is a random forest? Compare them and list 3 pros and cons of each?

# ## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
# 
# 1. Bagging and Boosting get N learners by generating additional data in the training stage. N new training data sets are produced by random sampling with replacement from the original set. By sampling with replacement some observations may be repeated in each new training data set. In the case of Bagging, any element has the same probability to appear in a new data set. However, for Boosting the observations are weighted and therefore some of them will take part in the new sets more often
# 
# 2. In decision trees, given a set of observations, the following question is asked: is every target variable in this set the same (or nearly the same)? If yes, label the set of observations with the most frequent class; if no, find the best rule that splits the observations into the purest set of observations. 
# 1. Easy to interpret and make for straightforward visualizations.
# 2. The internal workings are capable of being observed and thus make it possible to reproduce work.
# 3. Fast
# 
#     Random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction. 
# 1. Prevent the overfitting probem of DT.
# 2. Less variance
# 3. More accurate
#     

# ### 6) PCA vs Autoencoders
# - Describe how PCA achieves dimensionality reduction. Outline the main steps of the algorithm
# - What is the importance of eigenvectors and eigenvalues in the PCA algorithm above.
# - When we compute the covariance matrix in PCA, we have to subtract the mean. Why do we do this?
# - What is Autoencoder (compare it to PCA)? Why are autoencoders better in general.
# - When is the reduced dimension of an encoder equivalent to that of a PCA

# ## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
# 
# PCA reduces the number of features to be used in the final model by focusing only on the components accounting for the majority of the variance in the dataset and removes correlation between features.
# 1. Subtract the mean from each of the data dimensions. It makes sure that your end result is not dominated by a single variable. So we have to standardize the data set, as it is possible that different variables are measured in different scales.
# 2. Calculate the covariance matrix
# 3. Finding eigenvalues and eigenvectors. The eigenvalues tell us the variance in the data set and eigenvectors tell us the corresponding direction of the variance. 
# 4. Sort the eigenvalues and eigenvectors and select top k eigenvalues and corresponding eigenvectors
# 5. Forming the new data set in reduced dimensions
# 
# 
# PCA is restricted to a linear map, while auto encoders can have nonlinear enoder/decoders.
# Training an autoencoder with one dense encoder layer and one dense decoder layer and linear activation is essentially equivalent to performing PCA.
# 

# ### 7) Implementation
# 
# In the 1980's', Alex 'Sandy' Pentland came up with 'EigenFaces'. A novel way for facial classification using dimensionality reduction. We are going to try replicate the experiment in this question. We have loaded the face dataset for you below. Here's some steps for you: 
# 
# - Use PCA to reduce its dimensionality.
# - Use any algorithm to train a classifier for the dataset. You may use sklearn or pytorch. (Refer to PCA demo notebook for hints)
# - (Optional) Use autoencoders for the dimensionality reduction, compare results to PCA. Any comments/conculsions?
# 

# In[2]:


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


# In[3]:


import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# plot an example image
plt.imshow(X[1].reshape(h,w), cmap = 'gray')

### insert your code here ###

StdSc = StandardScaler()
X=StdSc.fit_transform(X)
pca = PCA(n_components=50)
X = pca.fit_transform(X)

trainx=X[:644]
trainy=y[:644]
testx=X[644:]
testy=y[644:]

def accuracy(target, predict):
    return (sum(target == predict) / float(target.size))*100

from sklearn.neighbors import KNeighborsClassifier as KNC
knn= KNC(n_neighbors=3)
knn=knn.fit(trainx,trainy)
pred=knn.predict(testx)
print("Accuracy: ", str(accuracy(testy, pred)))


# ## Bonus Challenge! (Optional)
# 
# This will take some time. However, trust that it is a rewarding experience. There will be a prize for whoever implements it correctly!
# 
# - Implement a feed forward neural network with back proprogation using stochastic gradient descent by hand. 
# - Use any dataset you want and test the accuracy

# In[4]:


### your code below ###

