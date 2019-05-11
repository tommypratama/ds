
# coding: utf-8

# # SVM Model for Image Classification
# ##### Using SVM to classify MNIST data - a set of images of hand-written digits

# In[1]:

import pandas as pd


# ### MNIST data set: 
# Text images of 28x28 pixels represented as flattened array of 784 pixels <br />
# Each pixel is represented by a pixel intensity value from 0-255
# 
# <b>Download Link: </b>https://www.kaggle.com/c/3004/download/train.csv

# In[2]:

mnist_data = pd.read_csv("../data/mnist/train.csv")
mnist_data.tail()


# #### Preparing our training and test data
# The pixel intensities are divided by 255 so that they're all between 0 and 1

# In[3]:

from sklearn.model_selection import train_test_split

features = mnist_data.columns[1:]
X = mnist_data[features]
Y = mnist_data['label']

X_train, X_test, Y_train, y_test = train_test_split(X/255., Y, test_size=0.1, random_state=0)


# #### Create an SVM classifier model
# * penalty can be L1 or L2
# * dual set to false since we have many more samples than features

# In[4]:

from sklearn.svm import LinearSVC

clf_svm = LinearSVC(penalty="l2", dual=False, tol=1e-5)
clf_svm.fit(X_train, Y_train)


# #### Calculate accuracy of the model against the test set

# In[8]:

from sklearn.metrics import accuracy_score

y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print ('SVM accuracy: ',acc_svm)


# ### Grid Search
# - A brute-force way to obtain the best parameters for the ML algorithm
# - Tries out all combinations of parameters specified in the "grid"
# - Returns combination of parameters with the highest accuracy score
# - Since it explores all combinations - this will take a long time

# In[5]:

from sklearn.model_selection import GridSearchCV

penalties = ['l1', 'l2']
tolerances = [1e-3, 1e-4, 1e-5]

param_grid = {'penalty': penalties, 'tol': tolerances}

grid_search = GridSearchCV(LinearSVC(dual=False), param_grid, cv=3)
grid_search.fit(X_train, Y_train)

grid_search.best_params_


# #### Plugging in the "best parameters" to redefine the model 

# In[6]:

clf_svm = LinearSVC(penalty="l1", dual=False, tol=1e-3)
clf_svm.fit(X_train, Y_train)


# In[9]:

y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print ('SVM accuracy: ',acc_svm)


# In[ ]:



