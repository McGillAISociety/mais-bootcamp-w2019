#!/usr/bin/env python
# coding: utf-8

# # Assignment 2

# In this assignment, we will start with utilizing Sci-Kit learn to implement a linear regression model similar to what we did in Assignment 1. Afterwards, we will be dropping Sci-Kit learning and implementing these algorithms from scratch without the use of machine learning libraries. While you would likely never have to implement your own linear regression algorithm from scratch in practice, such a skill is valuable to have as you progress further into the field and find many scenarios where you actually may need to perform such implementations manually. Additionally, implementing algorithms from scratch will help you better understand the underlying mathematics behind each model.     

# ## Import Libraries

# We will be using the following libraries for this homework assignment. For the questions requiring manual implementation, the pre-existing implementations from Sci-Kit Learn should *not* be used.

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


# ## Preparing Data

# The file named **dataset1.csv** includes data that was generated from an n-degree polynomial with some gaussian noise. The data has 2 columns - first column is the feature (input) and the second column is its label (output). The first step is to load the data and split them into training, validation, and test sets. A reminder that the purpose of each of the splitted sets are as follows:
# 
# * **Training Set**: The sample of data used to fit the model
# * **Validation Set**: The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters.
# * **Test Set**: The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.
# 
# In the section below, we load the csv file and split the data randomnly into 3 equal sets. 
# 
# *Note that in practice, we usually aim for around a 70-20-10 split for train, valid, and test respectively, but due to limited data in our case, we will do an even split in order to have sufficient data for evaluation* 

# In[18]:


# Load the data and split into 3 equal sets
data = pd.read_csv('dataset1.csv', header=None)
data = data.iloc[:, :-1]
train, valid, test = np.split(data, [int(.33*len(data)), int(.66*len(data))])

# We sort the data in order for plotting purposes later
train.sort_values(by=[0], inplace=True)
valid.sort_values(by=[0], inplace=True)
test.sort_values(by=[0], inplace=True)


# Let's take a look at what our data looks like

# In[19]:


plt.scatter(train[0], train[1], s=10)
plt.show()


# Let's apply a linear regression model using Sci-Kit learn and see what the results look like.

# In[20]:


# Reshape arrays since sci-kit learn only takes in 2D arrays
train_x = np.array(train[0])
train_y = np.array(train[1])
valid_x = np.array(valid[0])
valid_y = np.array(valid[1])
train_x = train_x.reshape(-1, 1)
train_y = train_y.reshape(-1, 1)
valid_x = valid_x.reshape(-1, 1)
valid_y = valid_y.reshape(-1, 1)

# Apply linear regression model
model = LinearRegression()
model.fit(train_x, train_y)
y_pred = model.predict(train_x)

# Plot the results
plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, y_pred, color='r')
plt.show()


# By analyzing the line of best fit above, we can see that a straight line is unable to capture the patterns of the data. This is an example of underfitting. As seen in the latest lecture, we can generate a higher order equation by adding powers of the original features as new features. 
# 
# The linear model,: 
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ** $y(x)$ = $w_1 x$ + $w_0$ ** 
# 
# can be transformed to a polynomial model such as:
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ** $y(x)$ = $w_2 x^2$ + $w_1 x$ + $w_0$ ** 
# 
# Note that this is still considered to be linear model as the coefficients/weights associated with the features are still linear. x<sup>2</sup> is only a feature. However the curve that we would be fitting in this case is quadratic in nature.
# 
# Below we show an example of a quadratic curve being fit to the data

# In[21]:


# Create polynomial features with degree 2
polynomial_features = PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(train_x)

# Apply linear regression
model = LinearRegression()
model.fit(x_poly, train_y)
y_poly_pred = model.predict(x_poly)

# Plot the results
plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, y_poly_pred, color='r')
plt.show()


# As you can see, we get a slightly better fit with a quadratic curve. Let's use the model to make predictions on our validation set and compute the mean squared error, which is the error which we wish to minimize.

# In[22]:


# Make predictions using pretrained model
valid_y_poly_pred = model.predict(polynomial_features.fit_transform(valid_x))

# Calculate root mean squared error
mse = mean_squared_error(valid_y, valid_y_poly_pred)
print("Mean Squared Error: {}".format(mse))

# Plot the prediction results
plt.scatter(valid_x, valid_y, s=10)
plt.plot(valid_x, valid_y_poly_pred, color='r')
plt.show()


# ## Question 1: Polynomial Regression Using Sci-Kit Learn
# 
# Now it is your turn! Following the same format as above, implement a 10-degree polynomial regression model on the training data and plot your results. Use your model to predict the output of the validation set and calculate the mean square error. Report and plot the results. 

# In[23]:


### YOUR CODE HERE - Fit a 10-degree polynomial using Sci-Kit Learn
polynomial_features = PolynomialFeatures(degree=10)
x_poly = polynomial_features.fit_transform(train_x)
model = LinearRegression()
model.fit(x_poly, train_y)
y_poly_pred = model.predict(x_poly)
### YOUR CODE HERE - Plot your the curve on the training data set
plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, y_poly_pred, color='r')
plt.show()
### YOUR CODE HERE - Use model to predict output of validation set
valid_y_poly_pred = model.predict(polynomial_features.fit_transform(valid_x))
### YOUR CODE HERE - Calculate the RMSE. Report and plot the curve on the validation set.
mse = mean_squared_error(valid_y, valid_y_poly_pred)
print("Mean Squared Error: {}".format(mse))

# Plot the prediction results
plt.scatter(valid_x, valid_y, s=10)
plt.plot(valid_x, valid_y_poly_pred, color='r')
plt.show()


# #### Did the root mean squared error go up or down as compared to the 2-degree polynomial curve? Why do you think this is the case?
# 
# The MSE went down with this higher degree curve. This is due to the fact that more details of the dataset can be "remembered", allowing for more accurate predictions.

# Now repeat the above for a 20-degree polynomial regression model.

# In[24]:


### YOUR CODE HERE - Fit a 20-degree polynomial using Sci-Kit Learn
polynomial_features = PolynomialFeatures(degree=20)
x_poly = polynomial_features.fit_transform(train_x)
model = LinearRegression()
model.fit(x_poly, train_y)
y_poly_pred = model.predict(x_poly)
### YOUR CODE HERE - Plot your the curve on the training data set
plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, y_poly_pred, color='r')
plt.show()
### YOUR CODE HERE - Use model to predict output of validation set
valid_y_poly_pred = model.predict(polynomial_features.fit_transform(valid_x))
### YOUR CODE HERE - Calculate the RMSE. Report and plot the curve on the validation set.
mse = mean_squared_error(valid_y, valid_y_poly_pred)
print("Mean Squared Error: {}".format(mse))

# Plot the prediction results
plt.scatter(valid_x, valid_y, s=10)
plt.plot(valid_x, valid_y_poly_pred, color='r')
plt.show()


# #### How does the mean square error compare to the previous two models? Why do you think this is the case?
# 
# The MSE for this model has gone way up. The curve is very accurate for the training data, but fails on the validation. This is a case of overfitting as the model has picked up too much of the noise and details of the training dataset.

# ## Question 2: Manual Implementation

# Now it's time to appreciate the hard work that open source developers have put, in order to allow you to implemenent machine learning models without doing any math! No more Sci-Kit learn (or any other libraries like Tensorflow, Pytorch, etc) for the rest of this assignment!

# Your first step is to fit a **10-degree polynomial** to the dataset we have been using above. Then using your results, calculate the mean squared error on both the training and validation set. You may use general utility libraries like numpy and pandas matrix computations and data manipulation, but pre-existing implementations of the model itself is prohibited.
# 
# A reminder that in polynomial regression, we are looking for a solution for the equation:
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ** $Y(X)$ = $W^T$ * $\phi(X)$ ** 
# 
# where
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ** $\phi(X)$ = [ $1$, $X$, $X^2$, $X^3$, ..... $X^n$ ] **
#  
# and the ordinary least square solution in closed form is:
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;** $W$ = $(X^T X)^{-1}X^TY$ **
# 
# Make sure to review the slides, do some research, and/or ask for clarification if this doesn't make sense. You must understand the underlying math before being able to implement this properly.
# 
# *Suggestion - Use the original pandas dataframes variables named train, valid, and test instead of the reshaped arrays that were used specifically for Sci-Kit Learn. It will make your computations cleaner and more inuitive.*

# In[25]:


### YOUR CODE HERE - Create the polynomial matrix ϕ(X)
#Transferring to NP arrays
xNP = np.array(train[0])
yNP = np.array(train[1])

#I realize this could likely be done directly into an NP array, but I was having some issues
#and this was working reliably
xPhi = []
for x in range(train[0].size):
    hold = []
    for n in range(11):
        hold.append(xNP[x]**n) #1 + x + x^2 + x^3 + ... + x^10
    xPhi.append(hold)

xPhiNP = np.array(xPhi)

### YOUR CODE HERE - Find the weighted matrix W
weight = np.matmul(np.linalg.inv(np.dot(xPhiNP.transpose(),xPhiNP)),np.dot(xPhiNP.transpose(),yNP))

### YOUR CODE HERE - Make predictions on the training set and calculate the root mean squared error. Plot the results.
#---------------------------------------------------------
#Predict y values through linear regression
def predict(xTest, weightMat):
    
    y = []
    
    #for every x, calculate the y given weighted matrix
    for x in xTest:
        prediction = 0
        for i in range(weightMat.size):
            prediction += weightMat.transpose()[i]*x**i
        y.append(prediction)
        
    return np.array(y)

#Calculate MSE
def MSE(yReal, yPred):
    diffSum = 0
    
    #for every calculated y, find diff between it and the real y and square it
    for i in range(yPred.size):
        diffSum += (yReal[i] - yPred[i])**2
        
    return diffSum/yReal.size
#----------------------------------------------------------

trainPred = predict(train[0], weight)
trainRMSE = MSE(train[1], trainPred)**(1/2) #RMSE

plt.scatter(train[0], train[1], s=10)
plt.plot(train[0], trainPred, color='r')
plt.show()

print("Train RMSE: {}".format(trainRMSE))

### YOUR CODE HERE - Make predictions on the validation set and calculate the mean squared error. Plot the results.
validPred = predict(valid[0], weight)

validMSE = MSE(np.array(valid[1]), validPred)
plt.scatter(valid[0], valid[1], s=10)
plt.plot(valid[0], validPred, color='r')
plt.show()

print("Valid MSE: {}".format(validMSE))


# For the rest of the assignment, we will use the other dataset named **dataset2.csv**. First load the csv and split the model into train, valid, and test sets as shown earlier in the assignment.

# In[26]:


### YOUR CODE HERE - Load dataset2.csv and split into 3 equal set
data2= pd.read_csv('dataset2.csv', header=None)
data2= data2.iloc[:, :-1]
train2, valid2, test2 = np.split(data2, [int(.33*len(data2)), int(.66*len(data2))])
### YOUR CODE HERE - Sort the data in order for plotting purposes later
train2.sort_values(by=[0], inplace=True)
valid2.sort_values(by=[0], inplace=True)
test2.sort_values(by=[0], inplace=True)


# Plot the data below to see what it looks like

# In[27]:


### YOUR CODE HERE - Plot the points for dataset2
plt.scatter(train2[0], train2[1], s=10)
plt.show()


# If done properly, you should see that the points fall under a relatively straight line with minor deviations. Looks like a perfect example to implement a linear regression model using the **gradient descent** method ..... without the use of any machine learning libraries!
# 
# Since the data falls along a straight line, we can assume the solution follows the form:
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ** $y(x)$ = $m x$ + $b$ **
# 
# A reminder that in gradient descent, we essentially want to iteratively get closer to the minimum of our objective function (the mean squared error), such that:
#  
#  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ** MSE($w_0$) > MSE($w_1$) > MSE($w_2$) > ...**
# 
# The algorithm is as follows:
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ** 1) Pick initial $w_0$ randomnly. **
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ** 2) For $k=1,2..$ $\Rightarrow$ $w_{k+1}$ = $w_k$ - $\alpha$  $g(w_k)$  where $\alpha > 0$ is the learning rate and $g(w_k)$ is the gradient. **
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ** End when | $w_k$ + $1$ - $w_k$ | < $\epsilon$ **
# 
# 
# Make sure to review the slides, do some research, and/or ask for clarification if this doesn't make sense. There are many resources online for gradient descent. You must understand the underlying math before being able to implement this properly.
# 
# Now once you understand, it is time to implement the gradient descent below. You may set the learning rate to 1e-6 or whatever value you think is best. As usual, calculate the mean squared error and plot your results. This time, training should be done using the training and validation sets, while the final mean squared error should be computed using the testing set.

# ## Turning In

# In[28]:


### YOUR CODE HERE - Implement gradient decent

#----------------------------------------------------------
def gradDesc(x, y, mCurr=0, bCurr=0, learnRate=0.001, precision=1e-8):
    #Setting N to be used in calculations and to avoid int devision
    N = float(len(y))
    
    #initializing var so that break check can be done in fist loop
    costPrev = 0
    
    #Equivalent to do-while loop, runs until cost change is below a set value
    while (True) :
        #Current predictions
        yPred = (mCurr * x) + bCurr
        
        sumSquared = 0
        
        #Cost calculation, MSE of prediction
        '''
        for val in (y-yPred):
            sumSquared += val**2
        
        costCurr = sumSquared / N
        '''
        costCurr = MSE(y, yPred)
        
        #Partial derivatives for slope and intercept, found online and inserted cause the most complicated
        #math I've done is Cal 2
        mGrad = -(2/N) * sum(x * (y - yPred))
        bGrad = -(2/N) * sum(y - yPred)
        
        #Trying to improve predictions in next step
        mCurr = mCurr - (learnRate * mGrad)
        bCurr = bCurr - (learnRate * bGrad)
        
        #Exits if change in cost is below set amount
        if (abs(costCurr - costPrev) < precision) : 
            break
            
        costPrev = costCurr

    return mCurr, bCurr, costCurr

#Generate predicted y values based on predicted equation
def genVals(xSet, m, b):
    y = []
    for x in xSet :
        y.append(m*x + b)
        
    return y
#----------------------------------------------------------

#Calculate variables for predicted line
m, b, cost = gradDesc(train2[0], train2[1], learnRate=0.001)

#Generate points from calculated line
trainPred2 = genVals(train2[0], m, b)
validPred2 = genVals(valid2[0], m, b)
testPred2 = genVals(test2[0], m, b)
testMSE = RMSE(np.array(test2[1]), np.array(testPred2))

#Print plot
plt.scatter(train2[0], train2[1], s=10)
plt.plot(train2[0], trainPred2, color='r')
plt.show()

#Print prediction info
print("Slope: {}".format(m))
print("Y-intercept: {}".format(b))
print("Cost: {}".format(cost))
print("\nEquation: y =", m, "* x +", b)

#Print plot
plt.scatter(valid2[0], valid2[1], s=10)
plt.plot(valid2[0], validPred2, color='r')
plt.show()

#Print plot
plt.scatter(test2[0], test2[1], s=10)
plt.plot(test2[0], testPred2, color='r')
plt.show()

print("MSE: {}".format(testMSE))


# 1. Convert this notebook to a regular python file (file -> download as -> python)
# 
# 2. Submit both the notebook and python file via a pull request as specified in the README

# In[ ]:




