
# coding: utf-8

# # Working with Categorical and Numeric Data
# ##### Using Label Encoding and One Hot Encoding for categorical data; Apply scaling to numeric data

# In[1]:

import pandas as pd


# In[2]:

print(pd.__version__)


# ### Sample data representing student data and exam scores
# Download link: http://roycekimmons.com/system/generate_data.php?dataset=exams&n=100

# In[4]:

exam_data = pd.read_csv('../data/exams.csv', quotechar='"')
exam_data


# #### Check out average score for each exam

# In[5]:

math_average = exam_data['math score'].mean()
reading_average = exam_data['reading score'].mean()
writing_average = average = exam_data['writing score'].mean()

print('Math Avg: ', math_average)
print('Reading Avg: ', reading_average)
print('Writing Avg: ', writing_average)


# ### Data Standardization:
# Apply scaling on the test scores to express them in terms of <b>z-score</b> <br />
# Z-score is the expression of a value in terms of the number of standard deviations from the mean <br />
# The effect is to give a score which is relative to the the distribution of values for that column

# In[6]:

from sklearn import preprocessing

exam_data[['math score']] = preprocessing.scale(exam_data[['math score']])
exam_data[['reading score']] = preprocessing.scale(exam_data[['reading score']])
exam_data[['writing score']] = preprocessing.scale(exam_data[['writing score']])


# In[7]:

exam_data


# #### Explore averages after scaling

# In[8]:

math_average = exam_data['math score'].mean()
reading_average = exam_data['reading score'].mean()
writing_average = average = exam_data['writing score'].mean()

print('Math Avg: ', math_average)
print('Reading Avg: ', reading_average)
print('Writing Avg: ', writing_average)


# ### Label Encoding:
# Convert text values to numbers. These can be used in the following situations:
# * There are only two values for a column in your data. The values will then become 0/1 - effectively a binary representation
# * The values have relationship with each other where comparisons are meaningful (e.g. low<medium<high)

# In[9]:

le = preprocessing.LabelEncoder()
exam_data['gender'] = le.fit_transform(exam_data['gender'].astype(str))


# In[10]:

exam_data.head()


# In[11]:

le.classes_


# ### One-Hot Encoding:
# * Use when there is no meaningful comparison between values in the column
# * Creates a new column for each unique value for the specified feature in the data set

# In[13]:

pd.get_dummies(exam_data['race/ethnicity'])


# #### Include the dummy columns in our data set

# In[15]:

exam_data = pd.get_dummies(exam_data, columns=['race/ethnicity'])


# In[12]:

exam_data


# #### Apply one-hot-encoding for remaining non-numeric features

# In[13]:

exam_data = pd.get_dummies(exam_data, columns=['parental level of education', 
                                               'lunch', 
                                               'test preparation course'])


# #### The data is now ready to be used to train a model

# In[14]:

exam_data.head()


# In[ ]:



