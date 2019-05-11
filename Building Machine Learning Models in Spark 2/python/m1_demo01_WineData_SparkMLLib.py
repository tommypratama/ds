
# coding: utf-8

# ### Check the Spark Context
# Spark context sets up internal services and establishes a connection to a Spark execution environment

# In[3]:


sc


# <b>Dataset location: </b>https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

# In[1]:


rawData = sc.textFile('../datasets/wine.data') 


# #### The raw data is of type MapPartitionsRDD
# MapPartitionsRDD is the result of the following transformations:
# * map
# * flatMap
# * filter
# * glom
# 
# MapPartitionsRDD is an RDD that applies the provided function f to every partition of the parent RDD

# In[2]:


rawData


# #### View contents of the rawData RDD

# In[4]:


rawData.take(10)


# #### Function to transform each row in the RDD to a LabeledPoint
# * MLlib classifiers and regressors require data sets in a format of rows of type LabeledPoint
# * It's in the format (&lt;label&gt;, [&lt;array_of_features&gt;])

# In[5]:


from pyspark.mllib.regression import LabeledPoint

def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[0], values[1:])


# #### Transform our raw data into an RDD of LabeledPoints

# In[6]:


parsedData = rawData.map(parsePoint)
parsedData


# In[9]:


parsedData.take(10)


# #### Split the RDD into training and test data sets

# In[10]:


(trainingData, testData) = parsedData.randomSplit([0.8, 0.2])


# In[11]:


trainingData


# In[12]:


trainingData.take(10)


# ### Create a Decision Tree model
# * <b>numClasses: </b>The number of labels. Since the labels in our dataset are 1,2 or 3 (rather than 0, 1, 2), we specify 4 rather than 3. Otherwise, it complains when it encounters a label of 3
# * <b>categoricalFeaturesInfo: </b>Specifies which features are categorical. None of the features in our dataset are
# * <b>impurity: </b>Can be <i>gini</i> or <i>entropy</i>
# * <b>maxDepth: </b>Maximum depth of the decision tree
# * <b>maxBins: </b>Number of bins used when discretizing continuous features. Increasing maxBins allows the algorithm to consider more split candidates and make fine-grained split decisions - at the cost of computation

# In[13]:


from pyspark.mllib.tree import DecisionTree, DecisionTreeModel

model = DecisionTree.trainClassifier(trainingData, 
                                     numClasses=4, 
                                     categoricalFeaturesInfo={},
                                     impurity='gini', 
                                     maxDepth=3, 
                                     maxBins=32)


# #### Use our model to make predictions with our test data

# In[17]:


predictions = model.predict(testData.map(lambda x: x.features))
predictions.take(5)


# #### Pair up the actual and predicted values into a tuple

# In[19]:


labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
labelsAndPredictions.take(5)


# #### Compare the actual and predicted values to get the accuracy of our model

# In[21]:


testAcc = labelsAndPredictions.filter(
    lambda lp: lp[0] == lp[1]).count() / float(testData.count())
print('Test Accuracy = ' + str(testAcc))


# #### Use MulticlassMetrics instead for model evaluation
# * MulticlassMetrics takes rows of (prediction, label) tuples as input
# * The model can be evaluated on multiple measures such as fMeasure, precision, recall

# In[20]:


from pyspark.mllib.evaluation import MulticlassMetrics

metrics = MulticlassMetrics(labelsAndPredictions)


# In[22]:


metrics.accuracy


# In[23]:


metrics.fMeasure()


# #### Measure precision when making a specific prediction
# Check accuracy when the predicted value is 2.0

# In[24]:


metrics.precision(2.0)


# #### Plot a confusion matrix
# * MulticlassMetrics also provides a confusion matrix

# In[19]:


metrics.confusionMatrix()


# #### The confusion matrix is easier to read when converted to an array

# In[20]:


metrics.confusionMatrix().toArray()


# #### View the Decision Tree model
# It is merely a collection of if-else statements

# In[21]:


print(model.toDebugString())


# ### Spark can also handle data sets in LIBSVM format
# The data is in this format: <br />
# &lt;label&gt; &lt;index1&gt;:&lt;value1&gt; &lt;index2&gt;:&lt;value2&gt; ... <br /><br />
# 
# The MLUtils class is required to load SVM data

# In[25]:


from pyspark.mllib.util import MLUtils


# <b>LibSVM dataset location: </b>https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/wine.scale

# In[26]:


libsvmData = MLUtils.loadLibSVMFile(sc, '../datasets/wine.scale')


# In[27]:


libsvmData


# In[30]:


libsvmData.take(5)


# In[31]:


(trainingData, testData) = libsvmData.randomSplit([0.8, 0.2])


# #### The model has the same parameters as the one created previously

# In[32]:


libsvmModel = DecisionTree.trainClassifier(trainingData, 
                                           numClasses=4, 
                                           categoricalFeaturesInfo={},
                                           impurity='gini', 
                                           maxDepth=5, 
                                           maxBins=32)


# In[33]:


predictions = libsvmModel.predict(testData.map(lambda x: x.features))


# In[34]:


metrics = MulticlassMetrics(labelsAndPredictions)


# In[35]:


metrics.accuracy


# In[36]:


metrics.confusionMatrix().toArray()


# In[37]:


print(model.toDebugString())

