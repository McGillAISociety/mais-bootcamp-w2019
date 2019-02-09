
# coding: utf-8

# # Assignment 2

# In this assignment, we will start with utilizing Sci-Kit learn to implement a linear regression model similar to what we did in Assignment 1. Afterwards, we will be dropping Sci-Kit learning and implementing these algorithms from scratch without the use of machine learning libraries. While you would likely never have to implement your own linear regression algorithm from scratch in practice, such a skill is valuable to have as you progress further into the field and find many scenarios where you actually may need to perform such implementations manually. Additionally, implementing algorithms from scratch will help you better understand the underlying mathematics behind each model.     

# ## Import Libraries

# We will be using the following libraries for this homework assignment. For the questions requiring manual implementation, the pre-existing implementations from Sci-Kit Learn should *not* be used.

# In[182]:


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

# In[183]:


# Load the data and split into 3 equal sets
data = pd.read_csv('dataset1.csv', header=None)
data = data.iloc[:, :-1]

#are you using the whole dataset for train data?
train, valid, test = np.split(data, [int(.33*len(data)), int(.66*len(data))])

# We sort the data in order for plotting purposes later
train.sort_values(by=[0], inplace=True)
valid.sort_values(by=[0], inplace=True)
test.sort_values(by=[0], inplace=True)
print(data)


# Let's take a look at what our data looks like

# In[184]:


plt.scatter(train[0], train[1], s=10)
plt.show()


# Let's apply a linear regression model using Sci-Kit learn and see what the results look like.

# In[185]:


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

# In[186]:


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

# In[187]:


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

# In[188]:


#Fit a 10-degree polynomial using Sci-Kit Learn
poly_feat_10 = PolynomialFeatures(degree=10)
x_poly_10 = poly_feat_10.fit_transform(train_x)
model.fit(x_poly_10,train_y)
y_poly_10_pred = model.predict(x_poly_10)

#Plot your the curve on the training data set
plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, y_poly_10_pred, color='r')
plt.show()

#Use model to predict output of validation set
valid_poly_10_pred = model.predict(poly_feat_10.fit_transform(valid_x))

#Calculate the RMSE. Report and plot the curve on the validation set.
mse_10 = mean_squared_error(valid_y,valid_poly_10_pred)
print("Mean Squared Error: {}".format(mse_10))

plt.scatter(valid_x, valid_y, s=10)
plt.plot(valid_x, valid_poly_10_pred, color='r')
plt.show()


# #### Did the root mean squared error go up or down as compared to the 2-degree polynomial curve? Why do you think this is the case?
# 
# The rmse went down compared to the 2-degree polynomial because of underfitting with the 2nd degree. The 10th degree was able to capture the data trend with more complexity as it is a higher order polynomial.

# Now repeat the above for a 20-degree polynomial regression model.

# In[189]:


#Fit a 10-degree polynomial using Sci-Kit Learn
poly_feat_20 = PolynomialFeatures(degree=20)
x_poly_20 = poly_feat_20.fit_transform(train_x)
model.fit(x_poly_20,train_y)
y_poly_20_pred = model.predict(x_poly_20)

#Plot your the curve on the training data set
plt.scatter(train_x, train_y, s=10)
plt.plot(train_x, y_poly_20_pred, color='r')
plt.show()

#Use model to predict output of validation set
valid_poly_20_pred = model.predict(poly_feat_20.fit_transform(valid_x))

#Calculate the RMSE. Report and plot the curve on the validation set.
mse_20 = mean_squared_error(valid_y,valid_poly_20_pred)
print("Mean Squared Error: {}".format(mse_20))

plt.scatter(valid_x, valid_y, s=10)
plt.plot(valid_x, valid_poly_20_pred, color='r')
plt.show()


# #### How does the mean square error compare to the previous two models? Why do you think this is the case?
# 
# RMSE went way up due to overfitting on the training set. The polynomial was capturing too many niche trends in the training set, which led to a failure on the validation set.

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

# In[198]:


#Find the polynomial matrix ϕ(X)
#creating the x array, then reshaping to get ϕX
x_eqtns = [np.power(train[0],k) for k in range(11)]
ϕX = (np.matrix.transpose(np.array(x_eqtns)))

#Find the weighted matrix W
ϕX_trans = np.matrix.transpose(ϕX)
#find 𝑊= (𝑋𝑇𝑋)−1 * 𝑋𝑇𝑌
W = np.matmul(np.matmul(np.linalg.inv((np.matmul(ϕX_trans,ϕX))),ϕX_trans),train[1])

#Make predictions on the training set and calculate the root mean squared error. Plot the results.
y_tr_pred = np.matmul(ϕX,np.matrix.transpose(W))

plt.scatter(train[0], train[1], s=10)
plt.plot(train[0], y_tr_pred, color='r')
plt.show()

mse = mean_squared_error(train[1],y_tr_pred)
print("Mean Squared Error: {}".format(mse))

#Make predictions on the validation set and calculate the mean squared error. Plot the results.
val_x_eqtns = [np.power(valid[0],k) for k in range(11)]
ϕX = (np.matrix.transpose(np.array(val_x_eqtns)))
y_val_pred = np.matmul(ϕX,np.matrix.transpose(W))

plt.scatter(valid[0], valid[1], s=10)
plt.plot(valid[0], y_val_pred, color='r')
plt.show()

mse = mean_squared_error(valid[1],y_val_pred)
print("Mean Squared Error: {}".format(mse))


# For the rest of the assignment, we will use the other dataset named **dataset2.csv**. First load the csv and split the model into train, valid, and test sets as shown earlier in the assignment.

# In[191]:


#Load dataset2.csv and split into 3 equal sets
data_2 = pd.read_csv('dataset2.csv', header=None)
data_2 = data_2.iloc[:, :-1]

#split the data
train_2, valid_2, test_2 = np.split(data_2, [int(.33*len(data_2)), int(.66*len(data_2))])

#Sort the data in order for plotting purposes later
train_2.sort_values(by=[0], inplace=True)
valid_2.sort_values(by=[0], inplace=True)
test_2.sort_values(by=[0], inplace=True)


# Plot the data below to see what it looks like

# In[192]:


#Plot the points for dataset2
plt.scatter(train_2[0], train_2[1], s=10)
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

# ### 2.a) Function for getting the weight from gradient descent

# In[193]:


def gradient_descent(X,y,w=[2,3],a=1e-6,eps=1e-5):
    '''INPUT: 
    X = array of size(n,) y = splice of the pandas df
    w = estimated weight matrix. initialized to random value
    a = learning rate
    eps = value you wish your error to converge to
    
    OUTPUT: a weight matrix of size (2,)'''
    #random value for the first turn of while loop
    prev_w = [0,0]
    X_trans = np.matrix.transpose(X)
    
    while abs(w[1] - prev_w[1])>eps:
        #with each turn of the loop, the previous w becomes the w from the last turn
        prev_w = w
        error = np.matmul(X_trans,(np.matmul(X,prev_w) - y))
        w = prev_w - a*error

    return(w)


# ### 2.b) Test and validation sets

# In[200]:


#1. Training Data. create an X matrix for the training set
#reshape for calculations
x = [np.power(train_2[0],k) for k in range(2)]
X = np.matrix.transpose(np.array(x))

w_train = gradient_descent(X,train_2[1])
train_preds = np.matmul(X,w_train)

plt.title("training set")
plt.scatter(train_2[0], train_2[1], s=10)
plt.plot(train_2[0],train_preds,'r')
plt.show()

mse = mean_squared_error(train_2[1],train_preds)
print("Mean Squared Error: {}".format(mse))

##2. Validation - fine tune the weight by decreasing learning rate
x_val = [np.power(valid_2[0],k) for k in range(2)]
X_val = np.matrix.transpose(np.array(x_val))

#set initial weight to w_train
w_val = gradient_descent(X_val,valid_2[1], w=w_train, a=1e-7)
valid_preds = np.matmul(X_val,w_val)

plt.title("validation set")
plt.scatter(valid_2[0], valid_2[1], s=10)
plt.plot(valid_2[0],valid_preds,'r')
plt.show()

mse = mean_squared_error(valid_2[1],valid_preds)
print("Mean Squared Error: {}".format(mse))


# ### 2.c) Now using the weight from the validation set, predict results on test set

# In[201]:


##3.Test using w_val
x_test = [np.power(test_2[0],k) for k in range(2)]
X_test = np.matrix.transpose(np.array(x_test))

test_preds = np.matmul(X_test,w_val)
plt.title("test set")
plt.scatter(test_2[0],test_2[1], s=10)
plt.plot(test_2[0],test_preds,'r')
plt.show()

mse = mean_squared_error(test_2[1],test_preds)
print("Mean Squared Error: {}".format(mse))


# ## Turning In

# 1. Convert this notebook to a regular python file (file -> download as -> python)
# 
# 2. Submit both the notebook and python file via a pull request as specified in the README
