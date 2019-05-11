
# coding: utf-8

# # Principal Components Analysis for Dimensionality Reduction
# ##### Create and test regression model before and after dimensionality reduction

# In[2]:

import pandas as pd
import numpy as np


# ### Download the Wine data set
# 
# <b>Download link: </b>https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
# 
# <b>Summary:</b> Given 11 features representing information about a number of white wines, predict its quality score
# 
# <b>Notes:</b>
# * The file comes with headers, but we specify them explicitly to be in our desired format
# * Since we're using our own headers, we skip the first row of the csv file which has the header
# 

# In[3]:

wine_data = pd.read_csv('../data/winequality-white.csv', 
                        names=['Fixed Acidity', 
                               'Volatile Acidity', 
                               'Citric Acid', 
                               'Residual Sugar', 
                               'Chlorides', 
                               'Free Sulfur Dioxide', 
                               'Total Sulfur Dioxide', 
                               'Density', 
                               'pH', 
                               'Sulphates', 
                               'Alcohol', 
                               'Quality'
                              ],
                        skiprows=1,
                        sep=r'\s*;\s*', engine='python')
wine_data.head()


# #### 7 Unique values. So wild guesses will be right about 14% of the time

# In[4]:

wine_data['Quality'].unique()


# ### Define training and test data
# Since all the data is already numeric, no conversions are necessary

# In[90]:

X = wine_data.drop('Quality', axis=1)
Y = wine_data['Quality']

from sklearn import preprocessing
X = preprocessing.scale(X)

from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# ### Define the benchmark SVM Classifier
# We check how our model works with all the features without any transformations

# In[150]:

from sklearn.svm import LinearSVC

clf_svc = LinearSVC(penalty='l1', dual=False, tol=1e-3)
clf_svc.fit(X_train, Y_train)


# #### Check the accuracy of the model

# In[151]:

accuracy = clf_svc.score(x_test, y_test)
print(accuracy)


# #### Plot a heatmap displaying the correlation between features

# In[152]:

import matplotlib.pyplot as plt
import seaborn as sns

corrmat = wine_data.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.set(font_scale=1.1)
sns.heatmap(corrmat, vmax=.8, square=True, annot=True, fmt='.2f', cmap = "winter")
plt.show()


# ### Use PCA for dimensionality reduction
# * <b>n_components: </b>Sets the number of dimensions
# * <b>whiten: </b>Before projecting the data to the principal components, the data will be normalized so that they have close to identity covariance. This has the effect of preventing one factor which has a high variance from being given too much importance

# In[173]:

from sklearn.decomposition import PCA

pca = PCA(n_components=1, whiten=True)
X_reduced = pca.fit_transform(X)


# #### View the eigen values of each principal component in decreasing order

# In[174]:

pca.explained_variance_


# #### Eigen values expressed as a ratio

# In[175]:

pca.explained_variance_ratio_


# #### Generating a Scree Plot
# Can be used to visualize the Explained Variance and eliminate 

# In[176]:

import matplotlib.pyplot as plt
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Dimension')
plt.ylabel('Explain Variance Ratio')
plt.show()


# In[177]:

X_train, x_test, Y_train, y_test = train_test_split(X_reduced, Y, test_size=0.2, random_state=0)
clf_svc_pca = LinearSVC(penalty='l1', dual=False, tol=1e-3)
clf_svc_pca.fit(X_train, Y_train)

accuracy = clf_svc_pca.score(x_test, y_test)
print(accuracy)


# In[ ]:



