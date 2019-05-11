
# coding: utf-8

# # Support Vector Regression
# ##### Using SVR to predict the MPG of vehicles

# In[2]:

import pandas as pd


# In[3]:

print(pd.__version__)


# ### Download Auto MPG data set
# <b>Download Link:</b> https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
# 
# <b>Summary:</b> Given 8 pieces of information (features) about a vehicle, predict its mileage
# 
# <b>Notes:</b>
# * The file does not come with headers, so we specify them explicitly

# In[4]:

auto_data = pd.read_csv('../data/auto-mpg.data', delim_whitespace = True, header=None,
                   names = ['mpg', 
                            'cylinders', 
                            'displacement', 
                            'horsepower', 
                            'weight', 
                            'acceleration',
                            'model', 
                            'origin', 
                            'car_name'])


# In[5]:

auto_data.head()


# #### Check if the car_name feature can be helpful

# In[6]:

len(auto_data['car_name'].unique())


# In[7]:

len(auto_data['car_name'])


# #### Drop the car_name feature from the data frame
# There are too many unique values for any pattern to be detected

# In[67]:

auto_data = auto_data.drop('car_name', axis=1)
auto_data.head()


# #### Converting a numeric value for origin to something more meaningful
# * The values 1,2,3 represent America, Europe and Asia respectively
# * This is the first step before we apply one-hot-encoding for this feature
# * Renaming will give us more meaningful column names after one-hot-encoding is applied

# In[68]:

auto_data['origin'] = auto_data['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
auto_data.head()


# #### Applying one-hot-encoding now will give us more meaningful column names

# In[69]:

auto_data = pd.get_dummies(auto_data, columns=['origin'])
auto_data.head()


# #### Convert missing values in data frame to NaN

# In[70]:

import numpy as np

auto_data = auto_data.replace('?', np.nan)


# #### Drop rows with missing values

# In[71]:

auto_data = auto_data.dropna()
auto_data


# ### Prepare training and test data
# * Define the feature vector (X) and label (Y)
# * Use train_test_split to create data subsets sets for training and validations 
# * The test_size parameter specifies the proportion of the data required for testing

# In[72]:

from sklearn.model_selection import train_test_split

X = auto_data.drop('mpg', axis=1)

# Taking the labels (mpg)
Y = auto_data['mpg']

# Spliting into 80% for training set and 20% for testing set so we can see our accuracy
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# #### Define the Regression model
# * Use scikit-learn's SVR model
# * We start off with a linear kernel and set the regularization variable (C) to 1.0
# * The model is trained with the training data

# In[73]:

from sklearn.svm import SVR
regression_model = SVR(kernel='linear', C=1.0)
regression_model.fit(X_train, Y_train)


# #### Check the coefficients for each of the features

# In[74]:

regression_model.coef_


# #### Get R-square value with training data

# In[75]:

regression_model.score(X_train, Y_train)


# #### Use matplotlib to view the coefficients as a histogram

# In[76]:

from pandas import Series
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

predictors = X_train.columns
coef = Series(regression_model.coef_[0],predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')


# #### Get predictions on test data

# In[77]:

from sklearn.metrics import mean_squared_error

y_predict = regression_model.predict(x_test)


# #### Compare the predicted and actual values of the MPG

# In[78]:

get_ipython().magic(u'pylab inline')
pylab.rcParams['figure.figsize'] = (15, 6)

plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('MPG')

plt.legend()
plt.show()


# #### Get R-square value of predictions on test data

# In[79]:

regression_model.score(x_test, y_test)


# #### Calculate Mean Square Error

# In[80]:

regression_model_mse = mean_squared_error(y_predict, y_test)
regression_model_mse


# #### Root of Mean Square Error to measure degree to which our prediction is off

# In[81]:

import math

math.sqrt(regression_model_mse)


# In[ ]:



