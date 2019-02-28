
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


### Fit a 10-degree polynomial using Sci-Kit Learn
# Create polynomial features with degree 10
polynomial_features = PolynomialFeatures(degree=10)
x_poly = polynomial_features.fit_transform(train_x)

# Apply linear regression
model = LinearRegression()
model.fit(x_poly, train_y)
y_poly_pred = model.predict(x_poly)

### Plot curve on the training data set
plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, y_poly_pred, color='r')
plt.show()

### Use model to predict output of validation set
valid_y_poly_pred = model.predict(polynomial_features.fit_transform(valid_x))

### Calculate the RMSE. Report and plot the curve on the validation set.
# Calculate root mean squared error
mse = mean_squared_error(valid_y, valid_y_poly_pred)
print("Mean Squared Error: {}".format(mse))

# Plot the prediction results
plt.scatter(valid_x, valid_y, s=10)
plt.plot(valid_x, valid_y_poly_pred, color='r')
plt.show()


# #### Did the root mean squared error go up or down as compared to the 2-degree polynomial curve? Why do you think this is the case?
# 
# It went down compared to the 2-degree polynomial curve. A 10 degree polynomial is a better fit for the data.

# Now repeat the above for a 20-degree polynomial regression model.

# In[8]:


### Fit a 10-degree polynomial using Sci-Kit Learn
# Create polynomial features with degree 10
polynomial_features = PolynomialFeatures(degree=20)
x_poly = polynomial_features.fit_transform(train_x)

# Apply linear regression
model = LinearRegression()
model.fit(x_poly, train_y)
y_poly_pred = model.predict(x_poly)

### Plot curve on the training data set
plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, y_poly_pred, color='r')
plt.show()

### Use model to predict output of validation set
valid_y_poly_pred = model.predict(polynomial_features.fit_transform(valid_x))

### Calculate the RMSE. Report and plot the curve on the validation set.
# Calculate root mean squared error
mse = mean_squared_error(valid_y, valid_y_poly_pred)
print("Mean Squared Error: {}".format(mse))

# Plot the prediction results
plt.scatter(valid_x, valid_y, s=10)
plt.plot(valid_x, valid_y_poly_pred, color='r')
plt.show()


# #### How does the mean square error compare to the previous two models? Why do you think this is the case?
# 
# It went up. 20 degrees is too high, the model is learning the gaussian noise in the training data and performs poorly on test data.

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


# TRAINING
x = np.array(train[0])
y = np.array(train[1])

X = np.transpose(np.array([x**n for n in range(0, 10)]))
W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)

# Predictions on training data
y_train_pred = np.matmul(X, np.transpose(W))

plt.scatter(x, y, s=10)
plt.plot(x, y_train_pred, color='r')
plt.show()

mse = mean_squared_error(y, y_train_pred)
print("Mean Squared Error: {}".format(mse))

# VALIDATING
x_test = np.array(test[0])
y_test = np.array(test[1])

X_test = np.transpose(np.array([x_test**n for n in range(0, 10)]))

# Predictions on validation data
y_test_pred = np.matmul(X_test, np.transpose(W))

plt.scatter(x_test, y_test, s=10)
plt.plot(x_test, y_test_pred, color='r')
plt.show()

mse = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error: {}".format(mse))


# For the rest of the assignment, we will use the other dataset named **dataset2.csv**. First load the csv and split the model into train, valid, and test sets as shown earlier in the assignment.

# In[10]:


### YOUR CODE HERE - Load dataset2.csv and split into 3 equal sets
data = pd.read_csv('dataset2.csv', header=None)
data = data.iloc[:, :-1]
train, valid, test = np.split(data, [int(.33*len(data)), int(.66*len(data))])

### YOUR CODE HERE - Sort the data in order for plotting purposes later
train.sort_values(by=[0], inplace=True)
valid.sort_values(by=[0], inplace=True)
test.sort_values(by=[0], inplace=True)


# Plot the data below to see what it looks like

# In[11]:


plt.scatter(train[0], train[1], s=10)
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

# In[13]:


### YOUR CODE HERE - Implement gradient decent
x_train = np.array(train[0])
y_train = np.array(train[1])

X = np.transpose(np.array([x_train**n for n in range(0, 2)]))

def g(w, X, y):
    return sum([xi * (np.matmul(np.transpose(w), xi) - yi) for xi, yi in zip(X, y)])
def MSE(w, X, y):
    return mean_squared_error(y, np.matmul(X, w))
def gradient_descent(X, y, learning_rate, epsilon):
    # Guess initial w0
    w0 = [1, 0]
    for i in range(0, 1000): #time out after 1000 iterations
        w = w0 - learning_rate * g(w0, X, y_train)
        if abs(MSE(w, X, y_train) - MSE(w0, X, y_train)) < epsilon:
            return w
        w0 = w

w = gradient_descent(X, y_train, 0.005, 0.00001)
y_train_pred = np.matmul(X, w)

plt.scatter(x_train, y_train, s=10)
plt.plot(x_train, y_train_pred, color='r')
plt.show()

mse = mean_squared_error(y_train, y_train_pred)
print("Mean Squared Error: {}".format(mse))

### YOUR CODE HERE - Calculate the the mean squared error and plot the results.
x_test = np.array(test[0])
y_test = np.array(test[1])

X = np.transpose(np.array([x_test**n for n in range(0, 2)]))
y_test_pred = np.matmul(X, w)

plt.scatter(x_test, y_test, s=10)
plt.plot(x_test, y_test_pred, color='r')
plt.show()

mse = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error: {}".format(mse))

print("Fit parameters: ", w)


# ## Turning In

# 1. Convert this notebook to a regular python file (file -> download as -> python)
# 
# 2. Submit both the notebook and python file via a pull request as specified in the README
