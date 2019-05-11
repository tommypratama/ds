
# coding: utf-8

# # Lasso and Ridge Regression
# ##### First use Linear Regression to predict automobile prices. Then apply Lasso and Ridge Regression models on the same data and compare results

# In[1]:

import pandas as pd


# In[2]:

print(pd.__version__)


# ### Download the Automobile data set
# <b>Download link:</b> https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
# 
# <b>Summary:</b> Predict the price of a vehicle given other information about it
# 
# <b>Parameters: </b> <br />
# 1st argument is the location of the file (not necessarily a csv file) <br />
# <b>sep</b> specifies the separator, which can also be expressed as a regular expression. Here we trim whitespaces around the commas<br />
# <b>engine</b> represents the parsing engine. The values are <i>c</i> and <i>python</i>. The C engine is marginally faster but Python may offer more features 

# In[3]:

auto_data = pd.read_csv('../data/imports-85.data', sep=r'\s*,\s*', engine='python')
auto_data


# #### Fill missing values with NaN

# In[4]:

import numpy as np

auto_data = auto_data.replace('?', np.nan)
auto_data.head()


# #### Information about numeric fields in our dataframe
# Note that the automobile price is not present

# In[5]:

auto_data.describe()


# #### Information about all fields in our dataframe

# In[6]:

auto_data.describe(include='all')


# ### Data Cleaning
# Also called data cleansing. Involves identifying incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing, modifying, or deleting the dirty or coarse data.

# #### What data type is price?

# In[7]:

auto_data['price'].describe()


# #### Convert the values in the price column to numeric values
# If conversion throws an error set to NaN (by setting errors='coerce')

# In[8]:

auto_data['price'] = pd.to_numeric(auto_data['price'], errors='coerce') 


# In[9]:

auto_data['price'].describe()


# #### Dropping a column which we deem unnecessary

# In[10]:

auto_data = auto_data.drop('normalized-losses', axis=1)
auto_data.head()


# In[11]:

auto_data.describe()


# #### Horsepower is also non-numeric...

# In[12]:

auto_data['horsepower'].describe()


# #### ...so this is also converted to a numeric value

# In[13]:

auto_data['horsepower'] = pd.to_numeric(auto_data['horsepower'], errors='coerce') 


# In[14]:

auto_data['horsepower'].describe()


# In[15]:

auto_data['num-of-cylinders'].describe()


# #### Since there are only 7 unique values, we can explicitly set the corresponding numeric values

# In[16]:

cylinders_dict = {'two': 2, 
                  'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}
auto_data['num-of-cylinders'].replace(cylinders_dict, inplace=True)

auto_data.head()


# #### All other non-numeric fields can be made into usable features by applying one-hot-encoding

# In[17]:

auto_data = pd.get_dummies(auto_data, 
                           columns=['make', 
                                    'fuel-type', 
                                    'aspiration', 
                                    'num-of-doors', 
                                    'body-style', 
                                    'drive-wheels', 
                                    'engine-location', 
                                    'engine-type', 
                                    'fuel-system'])
auto_data.head()


# #### Drop rows containing missing values

# In[18]:

auto_data = auto_data.dropna()
auto_data


# #### Verify that there are no null values in the data set

# In[19]:

auto_data[auto_data.isnull().any(axis=1)]


# ### Data Cleaning is now complete
# We can now use our data to build our models

# #### Create training and test data using train_test_split

# In[20]:

from sklearn.model_selection import train_test_split

X = auto_data.drop('price', axis=1)

# Taking the labels (price)
Y = auto_data['price']

# Spliting into 80% for training set and 20% for testing set so we can see our accuracy
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# #### Create a LinearRegression model with our training data

# In[21]:

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)


# #### Check R-square on training data

# In[22]:

linear_model.score(X_train, Y_train)


# #### View coefficients for each feature

# In[23]:

linear_model.coef_


# #### A better view of the coefficients
# List of features and their coefficients, ordered by coefficient value

# In[24]:

predictors = X_train.columns
coef = pd.Series(linear_model.coef_,predictors).sort_values()

print(coef)


# #### Make predictions on test data

# In[25]:

y_predict = linear_model.predict(x_test)


# #### Compare predicted and actual values of Price

# In[26]:

get_ipython().magic(u'pylab inline')
pylab.rcParams['figure.figsize'] = (15, 6)

plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('Price')

plt.legend()
plt.show()


# #### R-square score
# For our model, how well do the features describe the price?

# In[27]:

r_square = linear_model.score(x_test, y_test)
r_square


# #### Calculate Mean Square Error

# In[28]:

from sklearn.metrics import mean_squared_error

linear_model_mse = mean_squared_error(y_predict, y_test)
linear_model_mse


# #### Root of Mean Square Error

# In[29]:

import math

math.sqrt(linear_model_mse)


# ### Lasso Regression
# Cost Function: RSS + <b>&alpha;</b>*(sum of absolute values of coefficients)
# 
# RSS = Residual Sum of Squares
# 
# Larger values of <b>&alpha;</b> should result in smaller coefficients as the cost function needs to be minimized

# In[30]:

from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.5, normalize=True)
lasso_model.fit(X_train, Y_train)


# #### Check R-square on training data

# In[31]:

lasso_model.score(X_train, Y_train)


# #### Coefficients when using Lasso

# In[32]:

coef = pd.Series(lasso_model.coef_,predictors).sort_values()
print(coef)


# #### Make predictions on test data

# In[33]:

y_predict = lasso_model.predict(x_test)


# #### Compare predicted and actual values of Price

# In[34]:

get_ipython().magic(u'pylab inline')
pylab.rcParams['figure.figsize'] = (15, 6)

plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('Price')

plt.legend()
plt.show()


# #### Check R-square value on test data

# In[35]:

r_square = lasso_model.score(x_test, y_test)
r_square


# #### Is the root mean square error any better?

# In[36]:

lasso_model_mse = mean_squared_error(y_predict, y_test)
math.sqrt(lasso_model_mse)


# ### Ridge Regression
# Cost Function: RSS + <b>&alpha;</b>*(sum of squares of coefficients)
# 
# RSS = Residual Sum of Squares
# 
# Larger values of α should result in smaller coefficients as the cost function needs to be minimized
# 
# Ridge Regression penalizes large coefficients even more than Lasso as coefficients are squared in cost function

# In[37]:

from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=0.05, normalize=True)
ridge_model.fit(X_train, Y_train)


# #### Check R-square on training  data

# In[38]:

ridge_model.score(X_train, Y_train)


# #### Coefficients when using Ridge

# In[39]:

coef = pd.Series(ridge_model.coef_,predictors).sort_values()
print(coef)


# #### Make predictions on test data

# In[40]:

y_predict = ridge_model.predict(x_test)


# #### Compare predicted and actual values of Price

# In[41]:

get_ipython().magic(u'pylab inline')
pylab.rcParams['figure.figsize'] = (15, 6)

plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('Price')

plt.legend()
plt.show()


# #### Get R-square value for test data

# In[42]:

r_square = ridge_model.score(x_test, y_test)
r_square


# In[43]:

ridge_model_mse = mean_squared_error(y_predict, y_test)
math.sqrt(ridge_model_mse)


# ### Apply SVR on this data set

# In[44]:

from sklearn.svm import SVR

regression_model = SVR(kernel='linear', C=1.0)
regression_model.fit(X_train, Y_train)


# #### R-square on training data

# In[45]:

regression_model.score(X_train, Y_train)


# In[46]:

coef = pd.Series(regression_model.coef_[0], predictors).sort_values()
print(coef)


# In[47]:

y_predict = regression_model.predict(x_test)


# In[48]:

get_ipython().magic(u'pylab inline')
pylab.rcParams['figure.figsize'] = (15, 6)

plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('Price')

plt.legend()
plt.show()


# #### R-square on test data

# In[49]:

r_square = regression_model.score(x_test, y_test)
r_square


# In[50]:

regression_model_mse = mean_squared_error(y_predict, y_test)
math.sqrt(regression_model_mse)


# In[ ]:



