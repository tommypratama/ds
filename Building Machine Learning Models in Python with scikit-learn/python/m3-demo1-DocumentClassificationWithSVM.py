
# coding: utf-8

# # SVM Model for Text Classification
# ##### Using SVM model to classify text documents into subject categories

# ### Import the 20 New Groups data set from the scikit-learn library
# * Data comprises a number of emails, articles and other text documents
# * Each document falls into one of 20 categories of "News"
# * Use a classification model to predict the news category given the document text

# In[10]:

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)


# In[11]:

twenty_train.keys()


# #### View the first document in our data set

# In[12]:

print(twenty_train.data[0]) 


# #### View all the categories

# In[13]:

twenty_train.target_names


# #### The target is represented by numbers

# In[14]:

twenty_train.target


# #### Create a bag of words from our document list

# In[15]:

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape


# #### View the word counts for the first document

# In[16]:

print(X_train_counts[0])


# #### Get TF-IDF Weights using TfidfTransformer
# This is different from TfidfVectorizer:
# * TfidfVectorizer takes in a list of documents as input and produces a TF-IDF weighted bag of words
# * TfidfTransformer takes in a regular bag of words and creates a TF-IDF weighted bag of words
# * TfidfVectorizer == CountVectorizer + TfidfTransformer

# In[17]:

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# #### Viewing the TF-IDF weights for first document

# In[18]:

print(X_train_tfidf[0])


# #### Create a Linear Support Vector Classifier
# * penalty specifies whether to use L1 norm or L2 norm
#     * Like with Lasso and Ridge, choose whether to minimize sum of absolute values or sum of squares of coefficients
# * dual specifies whether to solve the primal or dual optimization problem
#     * A primal optimization problem (e.g. increase revenue) can have an equivalent dual problem (e.g. reduce costs) (this is a gross oversimplification - a lot of math needed to explain in detail)
#     * In our example, the primal optimization could be to maximize distance between our model and nearest points on either side of it. This will have a corresponding dual optimization problem
#     * scikit-learn recommends that dual=False when there are more samples than features (which is the case in this example)
# * tol represents a tolerance for the algorithm to consider when trying to maximize or minimize an ojective function
#     * if the model is within the tolerance of the maximum or minimum, it is not refined further

# In[19]:

from sklearn.svm import LinearSVC

clf_svc = LinearSVC(penalty="l2", dual=False, tol=1e-3)
clf_svc.fit(X_train_tfidf, twenty_train.target)


# #### Alternatively, a scikit-learn Pipeline can be used
# * Pipeline is a sequence of transformations with an estimator specified in the final step
# * The output of one transformation is passed as input to the next transformation
# * The pipeline returns a model of the type specified in the estimator
# * When the fit() method of the model is called with arguments, the arguments are passed through the transformation steps before actually being applied to the model
# 

# In[20]:

from sklearn.pipeline import Pipeline

clf_svc_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf',LinearSVC(penalty="l2", dual=False, tol=0.001))
])


# In our example:
# * we pass the document corpus and the labels to the pipeline classifier
# * The CountVectorizer takes the corpus and creates a bag of words
# * The TfidfTransformer takes the bag of words and produces a TF-IDF weighted bag
# * The LinearSVC model applies the fit method with the TF-IDF weighted bag and the labels

# In[21]:

clf_svc_pipeline.fit(twenty_train.data, twenty_train.target)


# #### Obtain the test data which we will use to make predictions

# In[22]:

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)


# #### Make the predictions using our classifier

# In[23]:

predicted = clf_svc_pipeline.predict(twenty_test.data)


# #### Compute the accuracy of the model
# Remember, there are 20 categories, so wild guesses will result in an accuracy of about 0.05

# In[24]:

from sklearn.metrics import accuracy_score

acc_svm = accuracy_score(twenty_test.target, predicted)


# In[25]:

acc_svm


# #### How good is our model if we just used the word counts without transforming to TF-IDF weights?

# In[29]:

clf_svc_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf',LinearSVC(penalty="l2", dual=False, tol=0.001))
])


# In[30]:

clf_svc_pipeline.fit(twenty_train.data, twenty_train.target)
predicted = clf_svc_pipeline.predict(twenty_test.data)

acc_svm = accuracy_score(twenty_test.target, predicted)
acc_svm


# In[ ]:



