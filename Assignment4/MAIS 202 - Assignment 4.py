
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
Linear regression is a technique used to determine a best fit line given a certain data set. 
It uses the representation y = aX + b in order to predict what y will be given a certain x.

Polynomial regression is very similar to linear regression, but in a higher order. It is used to map functions that are not linear.
It creates more features by creating X = x^0 + x^1 + x^2 + ... up to n degree. 

Underfitting is the case where the computer has not been given enough data to train and therefore cannot fully represent 
the relationship between X and y.
For polynomial regression, underfitting when happen when the degree is lower than the function it is trying to represent.

Overfitting is the situation where the machine understands the relationship between X and y to an unreasonable extent.
Finding the relationship between each X and each y, but not of the data set in general. 
For polynomial regression, overfitting when happen when the degree is higher than the function it is trying to represent


# ### 2) Logistic Regression vs. Linear SVM
# - Describe how logistic regression works (3 lines or less)
# - Describe how linear SVM works. Mention the role(s) of:
#     - support vectors
#     - margin
#     - slack variables
#     - kernels
# - Plot an example for SVM where the linear kernel is not enough to separate the data, but another kernel works

# ### YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC
# 
# Logistical regression is a model used for suprevised learning of classification problems. For each X, it assigns only two
# possible y. Either 0 or 1. This model follows the sigmoid function.
# 
# SVM is used for classification problems with more than 2 options. 
# It does so by finding a hyperplane that separates the classes apart. The hyperplane is defined by support vectors. 
# The margin is defined as the width of the hyperplane separating two classifications apart.
# However, in the case where two classes can not be perfectly separated, there are slack variables.
# Slack variables are used to allow variables that fall into the margin between two classifications to still be correctly identified
# Kernels are functions used to describe how certain Xs fall into the same y. So it finds the similarity between those Xs and 
# attributes it to a certain space so that those Xs are grouped together.
# 
# 
# 

# In[17]:


from sklearn.datasets.samples_generator import make_circles
import matplotlib.pyplot as plt

    
X, y = make_circles(100, factor=.1, noise=.1)


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn', edgecolor = 'blue')
plt.show()


# ### 3) Linear SVM vs k-NN
# - K-Nearest Neighbours is a popular unsupervised learning algorithm. Explain the difference between supervised and unsupervised learning?
# - K-NN is an example of a lazy learning algorithm. Why is it called so. What could be a use case? Justify using a lazy learning algorithm in that case.
# - Outline the main steps for the KNN algorithm. Use text, code, plots, diagrams, etc as necessary.  
# - Plot a example dataset which works in an SVM classification and not k-NN classification. Repeat for the reverse scenario.

# In[14]:


## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC

Supervised learning is when each feature set is associated to a label already. Therefore we already know what the result is.
Unsupervised learning is trying to find specific patterns in the data to perhaps predict what future data looks like.

Lazy learning is when the algorithm does not require a training period. The model trains as it is being used.
This is very useful for data sets that are continuosly being updated since old training would quickly be rendered obsolete
For example: recommending netflix videos. New netflix shows are constantly being released

The steps for a KNN algorithm are:
    1- calculate the distance between the selected item to be classified and the other points
    2- select the nearest k items
    3- find what classification the majority of those points have. Use that classification for the selected point.


    


# #An example where SVM works but not k-NN
# 
# ![image.png](attachment:image.png)

# In[18]:


#An example where k-nn works but no SVM
#https://pythonspot.com/k-nearest-neighbors/
#this generates a couple of errors but the graph still shows, so I guess its good enough :) 
import matplotlib
matplotlib.use('GTKAgg')
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
 
n_neighbors = 6
 
# import some data to play with
iris = datasets.load_iris()
 
# prepare data
X = iris.data[:, :2]
y = iris.target
h = .02
 
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])
 
# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)
 
# calculate min, max and limits
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
 
# predict class using data and kNN classifier
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
 
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
 
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)" % (n_neighbors))
plt.show()


# ### 4) K-NN Implementation
# - Implement the K-NN algorithm by hand (ie. Don't use the sklearn implementation).

# In[37]:


# Implement kNN by hand. It might be useful to store all distances in one array/list

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import operator 
import math

# loading dataset
iris = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# Preview dataset
iris_df.head()

data, test = np.split(iris_df, [int(0.75*len(iris_df))])
## YOUR CODE HERE
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)
# step 1: 
def knn(trainingSet, testInstance, k):
 
    distances = {}
    sort = {}
 
    length = testInstance.shape[1]
    
    #### Start of STEP 3
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        
        #### Start of STEP 3.1
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)

        distances[x] = dist[0]
        #### End of STEP 3.1
 
    #### Start of STEP 3.2
    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    #### End of STEP 3.2
 
    neighbors = []
    
    #### Start of STEP 3.3
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    #### End of STEP 3.3
    classVotes = {}
    
    #### Start of STEP 3.4
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
 
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #### End of STEP 3.4

    #### Start of STEP 3.5
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0], neighbors)
# step 2:

# step 3:
k = 3
testSet = [[5.1, 3.5, 1.4, 0.2]]
test = pd.DataFrame(testSet)
result, neigh = knn(data, test, k)
print(result)
# ...


# ### 5) Ensemble Methods
# - Explain bagging and boosting. Clearly illustrate the difference between these methods. When would you use either one?
# - What is a decision tree? What is a random forest? Compare them and list 3 pros and cons of each?

# In[ ]:


## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC

Bagging takes small samples of the total training data and applies some form of classification method on it. 
This is repeated for a set amount of subsamples. Afterward, all the predictions are grouped together to form the final prediction
Because there are multiple subsamples being taken, bagging reduces the variance of the data.

Boosting on the other hand, makes a prediction on the dataset, and rebuilds the model based on it's accuracy. 
Therefore, it keeps being updated while increase its accuracy. However, this could lead to overfitting.
Since boosting works on previous iterations of the model, it reduces the model's bias.

A decision tree is a model built on the whole dataset, where it splits based on features. 
As there are more and more splits, the model becomes more and more accurate, however it can quickly lead to overfitting.
Pros:
    -Easy to visualize and understand
    -Can handle both regression and classification
    -Require relatively little effort for data preparation
Cons:
    -Can quickly be overfitted if not careful (over-complex trees)
    -Small variations in data can cause big changes in the trees generated. (Variance is very high)
    -It is a greedy algorithm, so cannot guarantee that it has reached the globally optimal decision tree
    
A random forest is an aggregate of decision trees, where each tree takes a subsample of the data and splits on a subset of those features.
As the final result depends on the aggregation of all decision trees, there is less variance.'
Pros:
    -Trees are not corrolated
    -Reduces risk of overfitting
    -Less variance for the model
Cons:
    -Hard to visualize 
    -Large amount of treees can make the model slow and ineffective
    -Not many hyperparameters (is this a con)


# ### 6) PCA vs Autoencoders
# - Describe how PCA achieves dimensionality reduction. Outline the main steps of the algorithm
# - What is the importance of eigenvectors and eigenvalues in the PCA algorithm above.
# - When we compute the covariance matrix in PCA, we have to subtract the mean. Why do we do this?
# - What is Autoencoder (compare it to PCA)? Why are autoencoders better in general.
# - When is the reduced dimension of an encoder equivalent to that of a PCA

# In[18]:


## YOUR ANSWER HERE - YOUR MAY USE MARKDOWN, LATEX, CODE, DIAGRAMS, ETC

PCA finds the relationship between various features in the dataset and builds a linear combination of those features while
attempting to retain as much variance as possible in order to keep the individuality of each data point.
The steps to perform PCA are:
    1-Standardize the featureset by substracting the mean
    2-Calculate the covariance.
    3-Find eigenvectors of the covariance 
    4-Create matrix W, with each column being the eigenvectors of the covariance
    5-Select the most relevant components determined by cross-validation
    
Eigenvectors and eigenvalues are important in the PCA algorithm since they allow us to keep the highest variance in the dataset.

We substract the mean to standardize the featureset and center it.

Autoencoders are another way to achieve dimensionality reduction. They train neural networks to reduce the dimension of the dataset.
This is done by having a bottleneck layer, where the amount of nodes is less than the input and output layers.
The advantage of using autoencoders is that they establish relationships that are not linear.

The reduced dimmension of an encoder is equivalent to a PCA when only linear activation functions are employed in the neural network.



# ### 7) Implementation
# 
# In the 1980's', Alex 'Sandy' Pentland came up with 'EigenFaces'. A novel way for facial classification using dimensionality reduction. We are going to try replicate the experiment in this question. We have loaded the face dataset for you below. Here's some steps for you: 
# 
# - Use PCA to reduce its dimensionality.
# - Use any algorithm to train a classifier for the dataset. You may use sklearn or pytorch. (Refer to PCA demo notebook for hints)
# - (Optional) Use autoencoders for the dimensionality reduction, compare results to PCA. Any comments/conculsions?
# 

# In[38]:


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


# In[48]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


X = StandardScaler().fit_transform(X)

X = np.array([data.ravel() for data in X])


X_average_row = np.average(X, axis=0)
X_bar = X - X_average_row
sigma = X_bar.T.dot(X_bar)

eigenvalues, _ = np.linalg.eig(sigma)

plt.plot(np.arange(len(eigenvalues)), eigenvalues)
plt.show()

pca = PCA(n_components = 40)
principalcomp = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(principalcomp)


# In[51]:


# plot an example image
plt.imshow(X[1].reshape(h,w), cmap = 'gray')


# In[52]:



### insert your code here ###
plt.imshow(X_reconstructed[1].reshape(h,w), cmap = 'gray')


# In[66]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = train_test_split(X_reconstructed, y, test_size=0.25)

rfc = RandomForestClassifier(n_estimators = 100, min_samples_split= 2, max_features = 80)

rfc.fit(X_train, y_train)


print('test acc: ', accuracy_score(rfc.predict(X_test), y_test))


# ## Bonus Challenge! (Optional)
# 
# This will take some time. However, trust that it is a rewarding experience. There will be a prize for whoever implements it correctly!
# 
# - Implement a feed forward neural network with back proprogation using stochastic gradient descent by hand. 
# - Use any dataset you want and test the accuracy

# In[ ]:




### your code below ###

