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
# Linear regression is the process of determining what constants m and b allow a linear equation of the form y = mx + b to minimize the error across all training datapoints.
# 
# Polynomial regression performs pretty much the same task, however it deals with a polynomial with x's from degree 1 to n and n + 1 constants.
# 
# Overfitting occurs when a model starts to learn too many details of the data it is being trained on and makes predictions based off the noise in the data. The model essentially memorizes the training data but has low accuracy when it predicts with data it hasn't encountered yet. Underfitting is roughly the opposite, when a model is unable to represent enough of the variation/trends and so fails to predict accurately.

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
# Logisic regression uses a sigmoid or logistic function to make predictions where the outcome should be one of a discrete set of classes. This form of regression performs similar optimization to linear/polynomial, however the constants are used in an equation which generates a term for the sigmoid/logistic function.
# 
# Linear SVM determines the best way to separate data in n-dimensional (where n is the number of features) space using a hyperplane. The basic concept can be summarized in 2D space with a line separating different groups of data points; however as the data becomes more complicated, it is sometimes necessary to represent the data with an additional axis and separate the points in this dimension. In this higher dimension, the goal is to maximize the distance from the nearest point of each class to the separation plane (which should be equidistant to the nearest point of both). The distance from the nearest point of each class to the plane is the margin, thus the goal is to maximize the margin. The plane that is determined from this is defined relative to these nearest points, the vectos to these points are called "support vectors". Sometimes it isn't possible/reasonable to have all training points of a class on one side of a plane, in this case we allow for "slack variables" which are present within the margin or in the wrong class's region. We obviously want to minimize the presence of these. Finally, predictions are made using the optimized plane, in order to compare the n-dimensional points to a high-dimension plane we use a kernel which performs linear algebra to translate between the two spaces.
# 
# This data doesn't work with a linear kernel, but can be accurate separated with an RBF kernel.
# ![YXSMs.png](https://i.stack.imgur.com/YXSMs.png)

# ### 3) Linear SVM vs k-NN
# - K-Nearest Neighbours is a popular unsupervised learning algorithm. Explain the difference between supervised and unsupervised learning?
# - K-NN is an example of a lazy learning algorithm. Why is it called so. What could be a use case? Justify using a lazy learning algorithm in that case.
# - Outline the main steps for the KNN algorithm. Use text, code, plots, diagrams, etc as necessary.
# - Plot a example dataset which works in an SVM classification and not k-NN classification. Repeat for the reverse scenario.

# ## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
# 
# Supervised learning is when you train a model using data with known classes or values which can be used to influence the improvement of the model. Unsupervised learning identifies trends and patterns in data without any known classes or values.
# 
# A lazy learning algorithm like K-NN never actually learns a function which describes the output data, these simply "memorize" the dataset. 
# 
# When a k-NN model is given a datapoint, it runs through all the stored datapoints and calculates it's similarity to them. A group of k items with the greatest similarity are selected. From these items, the probability of each class is determined; ie the fraction of the set which is made up of a given class. The class with the greatest probability is assigned to the input data.
# 
# A K-NN model works better on the data below as separating multiple classes can be difficult with SVM.
# ![GoodForKnn.png](https://cdn-images-1.medium.com/max/1200/1*mqRILg7L9KkJjiKc9TwGVg.png)
# 
# An SVM model would work work well on the below data as the points of each class are heavily mixed and not grouped well, however distinct separation paths can be identified.
# ![GoodForSVM](https://gyires.inf.unideb.hu/GyBITT/06/images/svm_exp6-data.png)

# ### 4) k-NN Implementation
# - Implement thea k-NN algorithm by hand (ie. Don't use the sklearn implementation).

# In[45]:


'''
So I just kinda ran into a roadblock and ran outta time. I know this is entirely incomplete and largely incorrect, 
but I felt I should cut my losses and submit before the dealine. If completion of this portion is necessary to 
complete the course let me know and I will finish it as soon as possible
'''
# Implement kNN by hand. It might be useful to store all distances in one array/list

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import operator

# loading dataset
iris = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# Preview dataset
print(iris_df.head())
#xTrain, xTest, yTrain, yTest = train_test_split(iris_df.drop('target'), iris_df['target'], test_size=35, random_state=42)



## YOUR CODE HERE

def getEuclideanDist(dataA, dataB) :
    for valA, valB in dataA, dataB:
        distance += pow(valA - valB, 2)
        
    return np.sqrt(distance)

def findType(distances) :
    
    
    
    return pred

# step 1: 
distances = np.empty(0)

for data in testSet:
    distance = getEuclideanDist(data, trainSet)
    np.append(distances, (distance, ))
# step 2:
distances.sort()
# step 3:

for 

# ...


# ### 5) Ensemble Methods
# - Explain bagging and boosting. Clearly illustrate the difference between these methods. When would you use either one?
# - What is a decision tree? What is a random forest? Compare them and list 3 pros and cons of each?

# ## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
# 
# Bagging involves using different subsets of the training data (and subests of the features) to train multiple models and predict based on the aggregation of these models' predictions. Boosting also involves the training of multiple models, however this uses weighted data which indicates which features each model should focus on. Bagging is useful when a model is overfitting as it "smoothes" the predicted function as it decreases the effect of variance in the data. Boosting is useful when a model continously misclassifies certain data cue to larger trends in the dataset, it is good at combatting bias.
# 
# A decision tree is a tree (in the cs sense) where each internal node consists of a case and the leaves contain the class to assign a datapoint which reaches it. Random forests is essentially just using the decision tree model with bagging heavily applied.
# 
# DT
#   - Pros:
#     - Easy to understand/visualize
#     - Can handle numerical or categorical data
#     - Efficient on large datasets
#     
#   - Cons:
#     - Prone to overfitting
#     - Greedy (doesn't always find best option)
#     - Can't really extrapolate
#     
# RF
#   - Pros:
#     - Can handle numerical or categorical data
#     - Decrease overfitting
#     - Efficient on large datasets
#     
#   - Cons:
#     - Difficult to interpret visually
#     - Can still overfit on noisy data
#     - Can't really extrapolate
#     

# ### 6) PCA vs Autoencoders
# - Describe how PCA achieves dimensionality reduction. Outline the main steps of the algorithm
# - What is the importance of eigenvectors and eigenvalues in the PCA algorithm above.
# - When we compute the covariance matrix in PCA, we have to subtract the mean. Why do we do this?
# - What is Autoencoder (compare it to PCA)? Why are autoencoders better in general.
# - When is the reduced dimension of an encoder equivalent to that of a PCA

# ## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
# 
# The dimensionality reductiion begins with the calculation of the covariance matrix which gives insight into the relationship between features. We determine the size and direction of the variance in this matrix in the form of eigen values and eigen vectors. The features with the greatest variace are identified and the original dataset is transformed.
# 
# Eigen vectors and eigen values are what allow us to determine the variance in the covariance matrix.
# 
# By subtracting the mean we centre the data which makes the data easier to interpret and the covariance matrix easier to define.
# 
# Autoencoders are neural networks where the target values are equal to the inputs and so attempt to learn an approximate identity funtion (as opposed to a process to simply reduce the dimensionality of the dataset). Autoencoders can handle non-linear data.
# 
# If you force the autoencoer to provide a linear output, the reduced dimension will be equivalent to that of a PCA.

# ### 7) Implementation
# 
# In the 1980's', Alex 'Sandy' Pentland came up with 'EigenFaces'. A novel way for facial classification using dimensionality reduction. We are going to try replicate the experiment in this question. We have loaded the face dataset for you below. Here's some steps for you: 
# 
# - Use PCA to reduce its dimensionality.
# - Use any algorithm to train a classifier for the dataset. You may use sklearn or pytorch. (Refer to PCA demo notebook for hints)
# - (Optional) Use autoencoders for the dimensionality reduction, compare results to PCA. Any comments/conculsions?
# 

# In[6]:


# loading the faces dataset
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import image

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


# In[37]:


# plot an example image
plt.imshow(X[1].reshape(h,w), cmap = 'gray')

### insert your code here ###
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

pca = PCA(n_components=50)
X_resized = pca.fit_transform(X)

xTrain, xTest, yTrain, yTest = train_test_split(X_resized, y, test_size=35, random_state=42)

#neural net cause sure why not
nn = MLPClassifier(hidden_layer_sizes=(800,), solver='adam', verbose=1)
nn.fit(xTrain, yTrain)

print('train acc: ', accuracy_score(nn.predict(xTrain), yTrain))
print('test acc: ', accuracy_score(nn.predict(xTest), yTest))


# ## Bonus Challenge! (Optional)
# 
# This will take some time. However, trust that it is a rewarding experience. There will be a prize for whoever implements it correctly!
# 
# - Implement a feed forward neural network with back proprogation using stochastic gradient descent by hand. 
# - Use any dataset you want and test the accuracy

# In[36]:


### your code below ###


# In[ ]:




