
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
data = data.iloc[:, :-1]
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
print(train_x)
train_x = train_x.reshape(-1, 1)
print(train_x)
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
polynomial_features_10 = PolynomialFeatures(degree=10)
x_poly_10 = polynomial_features_10.fit_transform(train_x)

model = LinearRegression()
model.fit(x_poly_10, train_y)
y_poly_pred_10 = model.predict(x_poly_10)


### YOUR CODE HERE - Plot your the curve on the training data set

plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, y_poly_pred_10, color='r')
plt.show()

### YOUR CODE HERE - Use model to predict output of validation set
valid_y_poly_pred_10 = model.predict(polynomial_features_10.fit_transform(valid_x))

### YOUR CODE HERE - Calculate the RMSE. Report and plot the curve on the validation set.
mse_10 = mean_squared_error(valid_y, valid_y_poly_pred_10)
print("Mean Squared Error: {}".format(mse_10))

plt.scatter(valid_x, valid_y, s=10)
plt.plot(valid_x, valid_y_poly_pred_10, color='r')
plt.show()


# #### Did the root mean squared error go up or down as compared to the 2-degree polynomial curve? Why do you think this is the case?
# 
# The mean squared error went down compared to the 2-degree polynomial curve. This is because it is a more accurate representation of the complexity. Therefore it the 10-degree polynomial is a better fit, however, we still do not know if it is overfit or still underfit.

# Now repeat the above for a 20-degree polynomial regression model.

# In[8]:


### YOUR CODE HERE - Fit a 20-degree polynomial using Sci-Kit Learn
polynomial_features_20 = PolynomialFeatures(degree=20)
x_poly_20 = polynomial_features_20.fit_transform(train_x)

model = LinearRegression()
model.fit(x_poly_20, train_y)
y_poly_pred_20 = model.predict(x_poly_20)


### YOUR CODE HERE - Plot your the curve on the training data set

plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, y_poly_pred_20, color='r')
plt.show()

### YOUR CODE HERE - Use model to predict output of validation set
valid_y_poly_pred_20 = model.predict(polynomial_features_20.fit_transform(valid_x))

### YOUR CODE HERE - Calculate the RMSE. Report and plot the curve on the validation set.

mse_20 = mean_squared_error(valid_y, valid_y_poly_pred_20)
print("Mean Squared Error: {}".format(mse_20))

plt.scatter(valid_x, valid_y, s=10)
plt.plot(valid_x, valid_y_poly_pred_20, color='r')
plt.show()


# #### How does the mean square error compare to the previous two models? Why do you think this is the case?
# 
# The mean square error increase compared to the 10-degree model and the 2-degree model. We can see that at this point, the curve is being overfitted. However, it is still impossible to tell if the 10-degree model is being overfit or underfit, only that the 20-degree model is definitely an overfit.

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

# In[22]:



# # # Plot the results
# plt.scatter(train_x, train_y, s=10)
# # # plt.plot(train_x, y_pred, color='r')
# plt.show()
# print(train_x.shape)
# print(train_y.shape)
# print(train_x)

#Variables working with, train_x, train_y, valid_x, valid_y



### YOUR CODE HERE - Create the polynomial matrix ϕ(X)

def polynomialMatrix(X, degree = 2):
    rowNumber = np.size(X,0)  #Returns the number of rows.
    polyMatrix = np.ones((rowNumber,1), dtype=float)
    for x in range(1,degree+1):
        polyMatrix = np.c_[polyMatrix, np.power(X,x,dtype=float)]
    return polyMatrix


polyMatrix_train = polynomialMatrix(train_x,10)
print(polyMatrix_train.shape)



### YOUR CODE HERE - Find the weighted matrix W

# w = ((X^T)X)^(-1) * X^T * y
def weightMatrix(x, y):
    x_transpose = np.transpose(x)

    stepOne = np.dot(x_transpose, x)
    stepTwo = np.linalg.inv(stepOne)
    stepThree = np.dot(stepTwo, x_transpose)
    stepFour = np.dot(stepThree, y)
    print("----------")
    print(x.shape)
    print(x_transpose.shape)
    print(stepOne.shape)
    print(stepTwo.shape)
    print(stepThree.shape)
    print(stepFour.shape)
    print("------------")
    return stepFour

# weightedMatrix = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(polyMatrix_train),polyMatrix_train)),np.transpose(polyMatrix_train)),train_y)
# print(weightedMatrix)
weightedMatrix = weightMatrix(polyMatrix_train, train_y)
# print(weightedMatrix.shape)

print(polyMatrix_train.shape)


### YOUR CODE HERE - Make predictions on the training set and calculate the root mean squared error. Plot the results.

def rmse(predictions, targets):
    return (((predictions - targets) ** 2).mean())

#IT seems like the dimensions don't fit correctly, will inverse the matrix multiplication and not tranpose the weighted matrix
predicted_y = np.dot(polyMatrix_train, weightedMatrix)

#predicted_y = np.transpose(weightedMatrix) @ polyMatrix_train


plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, predicted_y, color='r')
plt.show()

print(rmse(predicted_y,train_y))


    

### YOUR CODE HERE - Make predictions on the validation set and calculate the mean squared error. Plot the results.

polyMatrix_valid = polynomialMatrix(valid_x,10)

predicted_y_validation = np.dot(polyMatrix_valid, weightedMatrix)

# predicted_y_validation = np.multiply((np.transpose(weightedMatrix)), polyMatrix_valid)

plt.scatter(valid_x, valid_y, s=10)
plt.plot(valid_x, predicted_y_validation, color='r')
plt.show()

print(rmse(predicted_y_validation,valid_y))


# For the rest of the assignment, we will use the other dataset named **dataset2.csv**. First load the csv and split the model into train, valid, and test sets as shown earlier in the assignment.

# In[10]:




### YOUR CODE HERE - Load dataset2.csv and split into 3 equal sets

data2 = pd.read_csv('dataset2.csv', header=None)
data2 = data2.iloc[:, :-1]
train2, valid2, test2 = np.split(data2, [int(.33*len(data2)), int(.66*len(data2))])

### YOUR CODE HERE - Sort the data in order for plotting purposes later

train2.sort_values(by=[0], inplace=True)
valid2.sort_values(by=[0], inplace=True)
test2.sort_values(by=[0], inplace=True)

train2_x = np.array(train2[0])
train2_y = np.array(train2[1])
valid2_x = np.array(valid2[0])
valid2_y = np.array(valid2[1])
test2_x = np.array(test2[0])
test2_y = np.array(test2[1])
train2_x = train2_x.reshape(-1, 1)
train2_y = train2_y.reshape(-1, 1)
valid2_x = valid2_x.reshape(-1, 1)
valid2_y = valid2_y.reshape(-1, 1)
test2_x = test2_x.reshape(-1, 1)
test2_y = test2_y.reshape(-1, 1)


# Plot the data below to see what it looks like

# In[11]:


### YOUR CODE HERE - Plot the points for dataset2

plt.scatter(train2_x, train2_y, s=10)
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

# In[42]:


### YOUR CODE HERE - Implement gradient decent


# def computeCost(X, y, theta):
#     inner = np.power(((X @ theta.T) - y), 2) 
#     return np.sum(inner) / (2 * len(X))

#def gradient(X,y, theta):

##
def gradient_descent(x,y):
    m = b = 0
    iterations = 5000
    n = len(x)
    learning_rate = 0.008  
    for i in range(iterations):
        y_predicted = m * x + b                                 #calculate the current y at each iteration with current m and current b 
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)]) #We use MSE function as a cost function: sum all the difference between y-y_predicted using a for loop, and for each of those values, take their square. Do the sum of all those values divided by the number of elements
        md = -(2/n)*sum(x*(y-y_predicted))                      #Partial derivative of m
        bd = -(2/n)*sum(y-y_predicted)                          #Partial derivative of b
        m = m - learning_rate * md                              #updating the value of m
        b = b - learning_rate * bd                              #updating the value of b
        
    return(m,b,float(cost))
    
m, b, mse = gradient_descent(train2_x,train2_y)

##
# def gradientDescent(X, y, theta, alpha):
#     cost = 5
#     while (cost > 1.403):
#         #theta = theta - ((alpha/len(X)) * np.sum((X @ theta.T - y) * X, axis=0, dtype=float))
#         theta = theta - alpha*math.derivate(theta)
#         cost = computeCost(X, y, theta)
#         print(cost)
#     return (theta)

# theta = np.ones((1,1))
# # theta = np.array([[1.0, 1.0]])
# theta = gradientDescent(train2_x, train2_y, theta, alpha)
# print(theta.shape)
# print(len(train2_x))
# print(train2_x.shape)
# print(train2_y.shape)
# print(valid2_x.shape)
# print(test2_x.shape)
# print(theta)

predicted_y_test = m* test2_x + b


### YOUR CODE HERE - Calculate the the mean squared error and plot the results.

plt.scatter(test2_x, test2_y, s=10)
plt.plot(test2_x, predicted_y_test, color='r')
plt.show()

print(rmse(predicted_y_test,test2_y))


# 1. Convert this notebook to a regular python file (file -> download as -> python)
# 
# 2. Submit both the notebook and python file via a pull request as specified in the README
