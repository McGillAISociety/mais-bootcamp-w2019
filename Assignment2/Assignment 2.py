#!/usr/bin/env python
# coding: utf-8

# # Assignment 2

# In this assignment, we will start with utilizing Sci-Kit learn to implement a linear regression model similar to what we did in Assignment 1. Afterwards, we will be dropping Sci-Kit learning and implementing these algorithms from scratch without the use of machine learning libraries. While you would likely never have to implement your own linear regression algorithm from scratch in practice, such a skill is valuable to have as you progress further into the field and find many scenarios where you actually may need to perform such implementations manually. Additionally, implementing algorithms from scratch will help you better understand the underlying mathematics behind each model.     

# ## Import Libraries

# We will be using the following libraries for this homework assignment. For the questions requiring manual implementation, the pre-existing implementations from Sci-Kit Learn should *not* be used.

# In[1]:


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

# In[2]:


# Load the data and split into 3 equal sets
data = pd.read_csv('dataset1.csv', header=None)
data = data.iloc[:, :-1] #Purely integer-location based indexing for selection by position (x location all, y till -1)
train, valid, test = np.split(data, [int(.33*len(data)), int(.66*len(data))])

# We sort the data in order for plotting purposes later
train.sort_values(by=[0], inplace=True)
valid.sort_values(by=[0], inplace=True)
test.sort_values(by=[0], inplace=True)


# Let's take a look at what our data looks like

# In[3]:


plt.scatter(train[0], train[1], s=10)
plt.show()


# Let's apply a linear regression model using Sci-Kit learn and see what the results look like.

# In[4]:


# Reshape arrays since sci-kit learn only takes in 2D arrays
train_x = np.array(train[0])
train_y = np.array(train[1])
valid_x = np.array(valid[0])
valid_y = np.array(valid[1])

train_x = train_x.reshape(-1, 1) #What does this do? From an array of (165,) to (165, 1)
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

# In[5]:


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

# In[6]:


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

# In[7]:


### YOUR CODE HERE - Fit a 10-degree polynomial using Sci-Kit Learn

polynomial_ten = PolynomialFeatures(degree=10)
x_ten = polynomial_ten.fit_transform(train_x)
model.fit(x_ten, train_y)
y_ten = model.predict(x_ten)

### YOUR CODE HERE - Plot your the curve on the training data set

plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, y_ten, color='r')
plt.show()

### YOUR CODE HERE - Use model to predict output of validation set
valid_y_ten = model.predict(polynomial_ten.fit_transform(valid_x))
mse_ten = mean_squared_error(valid_y, valid_y_ten)
print("Mean Squared Error: {}".format(mse_ten))


### YOUR CODE HERE - Calculate the RMSE. Report and plot the curve on the validation set.
plt.scatter(valid_x, valid_y, s=10)
plt.plot(valid_x, valid_y_ten, color='r')
plt.show()


# #### Did the root mean squared error go up or down as compared to the 2-degree polynomial curve? Why do you think this is the case?
# 
# The root mean sqaured error go down when compared to the 2-degree polynomial curve because as you increase the degree of the polynomial, you increase its flexibility (it becomes more complex and can take into account more points (Slide 20 -25)). However, making the value of the degree very high overfits the data and you don't learn anything from your model.
# 

# Now repeat the above for a 20-degree polynomial regression model.

# In[8]:


### YOUR CODE HERE - Fit a 20-degree polynomial using Sci-Kit Learn
polynomial_twenty = PolynomialFeatures(degree=20)
x_twenty = polynomial_twenty.fit_transform(train_x)
model.fit(x_twenty, train_y)
y_twenty = model.predict(x_twenty)

### YOUR CODE HERE - Plot your the curve on the training data set

plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, y_twenty, color='r')
plt.show()

### YOUR CODE HERE - Use model to predict output of validation set
valid_y_twenty = model.predict(polynomial_twenty.fit_transform(valid_x))
mse_twenty = mean_squared_error(valid_y, valid_y_twenty)
print("Mean Squared Error: {}".format(mse_twenty))


### YOUR CODE HERE - Calculate the RMSE. Report and plot the curve on the validation set.
plt.scatter(valid_x, valid_y, s=10)
plt.plot(valid_x, valid_y_twenty, color='r')
plt.show()


# #### How does the mean square error compare to the previous two models? Why do you think this is the case?
# 
# The mean square root is higher than the one calculated in the previous 2 examples. As stated above, if we increase the degree of the polynomial too much, we stop learning the trends of a model. In the example above, we overfitted the data.

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

# In[9]:


def lin_regression_manual(degree, datasetX, datasetY):
    deg = np.arange((degree+1)) #array of powers
    a = np.full((len(datasetX), len(deg)), datasetX) #fill the dataset with values given by datasetX
    # ϕ(X) = (1, X_i X_i^2, X_i^3, ... , X_i^10)
    phi_x = np.power(a, deg) #each column = column ^ column-index
    W = np.matmul(np.linalg.inv(np.matmul(phi_x.transpose(), phi_x)), np.matmul(phi_x.transpose(), datasetY)) #closed-form solution formula
    fit_deg = np.poly1d(np.flip(W.ravel().transpose())) #array of coefficient saved in polynomial that is returned
    return fit_deg 


# In[10]:


def mean_squared_error(datasetY, polynomial_guess, datasetX):
    total_sum = 0
    for i in range(1, len(datasetY)):
        total_sum = total_sum + pow(datasetY.item(i) - np.polyval(polynomial_guess, datasetX.item(i)), 2)
    mse = total_sum/len(datasetY)
    return mse


# In[11]:


fit_10 = lin_regression_manual(10, train_x, train_y)
rmse = pow(mean_squared_error(train_y, fit_10, train_x), 0.5)
print("RMSE: ", rmse)

plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, np.polyval(fit_10, train_x), color='r')
plt.show()


#fit_10 = lin_regression_manual(10, valid_x, valid_y)
rms_error = pow(mean_squared_error(valid_y, fit_10, valid_x),0.5)
print("RMSE: ", rms_error)

plt.scatter(valid_x, valid_y, s=10)
plt.plot(valid_x, np.polyval(fit_10, valid_x), color='r')
plt.show()


# For the rest of the assignment, we will use the other dataset named **dataset2.csv**. First load the csv and split the model into train, valid, and test sets as shown earlier in the assignment.

# In[12]:


### YOUR CODE HERE - Load dataset2.csv and split into 3 equal sets
data = pd.read_csv('dataset2.csv', header=None)
data = data.iloc[:, :-1] #Purely integer-location based indexing for selection by position (x location all, y till -1)
train, valid, test = np.split(data, [int(.7*len(data)), int(.7*len(data))]) #a bad way of setting validation set to 0

### YOUR CODE HERE - Sort the data in order for plotting purposes later
train.sort_values(by=[0], inplace=True)
test.sort_values(by=[0], inplace=True)


# Plot the data below to see what it looks like

# In[13]:


### YOUR CODE HERE - Plot the points for dataset2
plt.scatter(train[0], train[1], s=10)
plt.show()

plt.scatter(test[0], test[1], s=10)
plt.show()


# If done properly, you should see that the points fall under a relatively straight line with minor deviations. Looks like a perfect example to implement a linear regression model using the **gradient descent** method ..... without the use of any machine learning libraries! 
# 
# **Question: Why is it the perfect example to do linear regression by gradient descent and not by calculating coefficients like before?**
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

# In[14]:


### YOUR CODE HERE - Implement gradient decent
train_x = np.array(train[0])
train_y = np.array(train[1])

test_x = np.array(test[0])
test_y = np.array(test[1])

train_x = train_x.reshape((-1, 1))
train_y = train_y.reshape(-1, 1)

test_x = test_x.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)

column_one_train = np.ones((len(train_x),1))
column_one_test = np.ones((len(test_x),1))


train_x = np.append(column_one_train, train_x, axis=1) #Added a column of 1
test_x = np.append(column_one_test, test_x, axis=1) #Added a column of 1


# In[15]:


LEARNING_RATE = 1e-4
THRESHOLD = 1e-6

def gradient_descent(X, y, w, alpha=LEARNING_RATE, iterations=10000):
    for _ in range(iterations):
        w = w - alpha * (np.matmul(np.matmul(np.transpose(X), X), w) - np.matmul(np.transpose(X), y))
    return w


# In[16]:


#Training weights
weights_train = gradient_descent(train_x, train_y, np.array([[1], [0]]))
print(weights_train)


# In[17]:


fit = np.poly1d(weights_train[::-1].ravel())
print(fit)

plt.scatter(train[0], train[1], s=10)
plt.plot(train_x, np.polyval(fit, train_x), color='r')
plt.show()


# In[18]:


mse = mean_squared_error(test_y, fit, test_x)
print("MSE = ", mse)

#Test set - fit obtained from training set tested on test set
plt.scatter(test[0], test[1], s=10)
plt.plot(test_x, np.polyval(fit, test_x), color='r')
plt.show()


# ## Turning In

# 1. Convert this notebook to a regular python file (file -> download as -> python)
# 
# 2. Submit both the notebook and python file via a pull request as specified in the README
