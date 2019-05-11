
# coding: utf-8

# # Extracting Features from Text
# ##### Using Bag of Words, TF-IDF Transformation

# In[1]:

from sklearn.feature_extraction.text import CountVectorizer


# #### Define a corpus of 4 documents with some repeated values

# In[2]:

corpus = ['This is the first document.',
          'This is the second document.', 
          'Third document. Document number three', 
          'Number four. To repeat, number four']


# #### Use CountVectorizer to convert a collection of text documents to a "bag of words"

# In[3]:

vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(corpus)

bag_of_words


# #### View what the "bag" looks like

# In[4]:

print(bag_of_words)


# #### Get the value to which a word is mapped

# In[5]:

vectorizer.vocabulary_.get('document')


# In[6]:

vectorizer.vocabulary_


# In[7]:

import pandas as pd

print(pd.__version__)


# In[8]:

pd.DataFrame(bag_of_words.toarray(), columns=vectorizer.get_feature_names())


# #### Extend bag of words with TF-IDF weights

# In[9]:

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
bag_of_words = vectorizer.fit_transform(corpus)

print(bag_of_words)


# In[10]:

vectorizer.vocabulary_.get('document')


# In[11]:

pd.DataFrame(bag_of_words.toarray(), columns=vectorizer.get_feature_names())


# #### View all the words and their corresponding values

# In[12]:

vectorizer.vocabulary_


# ### Hashing Vectorizer
# * One issue with CountVectorizer and TF-IDF Vectorizer is that the number of features can get very large if the vocabulary is very large
# * The whole vocabulary will be stored in memory, and this may end up taking a lot of space
# * With Hashing Vectorizer, one can limit the number of features, let's say to a number <b>n</b>
# * Each word will be hashed to one of the n values
# * There will collisions where different words will be hashed to the same value
# * In many instances, peformance does not really suffer in spite of the collisions

# In[13]:

from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(n_features=8)
feature_vector = vectorizer.fit_transform(corpus)
print(feature_vector)


# #### There is no way to compute the inverse transform to get the words from the hashed value
