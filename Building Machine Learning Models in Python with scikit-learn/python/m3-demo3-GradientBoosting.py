
# coding: utf-8

# # Gradient Boost Model for Regression
# ##### Using Gradient Boosting to predict the price of an automobile

# In[118]:

import pandas as pd


# ### Download the Automobile data set
# <b>Download Link</b>https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
# 
# <b>Summary:</b> Predict the price of a vehicle given other information about it

# In[119]:

auto_data = pd.read_csv('../data/imports-85.data', sep=r'\s*,\s*', engine='python')
auto_data


# #### Fill missing values with NaN

# In[120]:

import numpy as np

auto_data = auto_data.replace('?', np.nan)
auto_data.head()


# #### Information about numeric fields in our dataframe

# In[121]:

auto_data.describe()


# #### Information about all fields in our dataframe

# In[122]:

auto_data.describe(include='all')


# #### What data type is price?

# In[123]:

auto_data['price'].describe()


# #### Convert the values in the price column to numeric values
# If conversion throws an error set to NaN (by setting errors='coerce')

# In[124]:

auto_data['price'] = pd.to_numeric(auto_data['price'], errors='coerce') 


# In[125]:

auto_data['price'].describe()


# #### Dropping a column which we deem unnecessary

# In[126]:

auto_data = auto_data.drop('normalized-losses', axis=1)
auto_data.head()


# In[127]:

auto_data.describe()


# #### Horsepower is also non-numeric...

# In[128]:

auto_data['horsepower'].describe()


# #### ...so this is also converted to a numeric value

# In[129]:

auto_data['horsepower'] = pd.to_numeric(auto_data['horsepower'], errors='coerce') 


# In[130]:

auto_data['horsepower'].describe()


# In[131]:

auto_data['num-of-cylinders'].describe()


# #### Since there are only 7 unique values, we can explicitly set the corresponding numeric values

# In[132]:

cylinders_dict = {'two': 2, 
                  'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}
auto_data['num-of-cylinders'].replace(cylinders_dict, inplace=True)

auto_data.head()


# #### All other non-numeric fields can be made into usable features by applying one-hot-encoding

# In[133]:

auto_data = pd.get_dummies(auto_data, 
                           columns=['make', 'fuel-type', 'aspiration', 'num-of-doors', 
                                    'body-style', 'drive-wheels', 'engine-location', 
                                   'engine-type', 'fuel-system'])
auto_data.head()


# #### Drop rows containing missing values

# In[134]:

auto_data = auto_data.dropna()
auto_data


# #### Verify that there are no null values in the data set

# In[135]:

auto_data[auto_data.isnull().any(axis=1)]


# #### Create training and test data using train_test_split

# In[136]:

from sklearn.model_selection import train_test_split

X = auto_data.drop('price', axis=1)

# Taking the labels (price)
Y = auto_data['price']

# Spliting into 80% for training set and 20% for testing set so we can see our accuracy
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# ### Gradient Boosting:
# * Start off by learning a very simple model
# * Take the error residuals from the first model and then try to predict the errors in the next iteration (also with a simple learner)
# * Combine the two simple models to obtain a slightly better overall model
# * At each iteration, the learner tries to reduce the errors (not eliminate it) by a certain learning rate. This is also the gradient of the model
# * Keep iterating over the error residuals until you have an ensemble of simple learners which combine to produce a more complex model

# #### Parameters:
# - <b>n_estimators:</b> Number of boosting stages
# - <b>max_depth:</b> Maximum depth of each estimator tree
# - <b>min_samples_split: </b>Minimum samples in each subset when splitting the data set
# - <b>learning_rate: </b>Defines the rate at which to converge to the optimal value
# - <b>loss: </b>Type of loss function to optimize (ls == least squares)

# In[137]:

from sklearn.ensemble import GradientBoostingRegressor

params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gbr_model = GradientBoostingRegressor(**params)
gbr_model.fit(X_train, Y_train)


# #### Get R-square on training data

# In[138]:

gbr_model.score(X_train, Y_train)


# #### Make predictions on test data and compare with actual values

# In[139]:

y_predict = gbr_model.predict(x_test)


# In[140]:

get_ipython().magic(u'pylab inline')
pylab.rcParams['figure.figsize'] = (15, 6)

plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('MPG')

plt.legend()
plt.show()


# #### Calculate R-square score on test data

# In[141]:

r_square = gbr_model.score(x_test, y_test)
r_square


# #### Calculate Mean Square Error

# In[142]:

from sklearn.metrics import mean_squared_error

gbr_model_mse = mean_squared_error(y_predict, y_test)
gbr_model_mse


# #### Root of Mean Square Error

# In[143]:

import math

math.sqrt(gbr_model_mse)


# ### num_estimators vs learning_rate:
# - Higher learning rate should result in convergence with fewer estimators
# - High value for learning rate risks skipping the optimal solution
# - Low learning rate equates to high bias, high rate to high variance
# - Need to strike the right balance between num_estimators and learning_rate

# In[144]:

from sklearn.model_selection import GridSearchCV

num_estimators = [100, 200, 500]
learn_rates = [0.01, 0.02, 0.05, 0.1]
max_depths = [4, 6, 8]

param_grid = {'n_estimators': num_estimators,
              'learning_rate': learn_rates,
              'max_depth': max_depths}

grid_search = GridSearchCV(GradientBoostingRegressor(min_samples_split=2, loss='ls'),
                           param_grid, cv=3, return_train_score=True)
grid_search.fit(X_train, Y_train)

grid_search.best_params_


# #### Analyze the results of the grid search

# In[145]:

grid_search.cv_results_


# #### Extract the useful values from the Grid Search results

# In[146]:

for i in range(36):
    print('Parameters: ', grid_search.cv_results_['params'][i])
    print('Mean Test Score: ', grid_search.cv_results_['mean_test_score'][i])
    print('Rank: ', grid_search.cv_results_['rank_test_score'][i])
    print()


# In[147]:

params = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.05, 'loss': 'ls'}
gbr_model = GradientBoostingRegressor(**params)
gbr_model.fit(X_train, Y_train)


# #### Compare predictions vs actual values

# In[148]:

y_predict = gbr_model.predict(x_test)


# In[149]:

get_ipython().magic(u'pylab inline')
pylab.rcParams['figure.figsize'] = (15, 6)

plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('MPG')

plt.legend()
plt.show()


# #### R-square on test data

# In[150]:

r_square = gbr_model.score(x_test, y_test)
r_square


# In[151]:

gbr_model_mse = mean_squared_error(y_predict, y_test)
math.sqrt(gbr_model_mse)


# In[ ]:



