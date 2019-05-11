
# coding: utf-8

# ### Check the SQLContext
# The entry point into all functionality in Spark SQL <br />
# The SQLContext can be used to create a Spark Dataframe (as opposed to an RDD) from a data source

# <b>Dataset location: </b>https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data <br />
# The same dataset we have used so far

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName('Predicting the grape variety from wine characteristics')     .getOrCreate()

rawData = spark.read            .format('csv')            .option('header', 'false')            .load('../datasets/wine.data')


# #### View the schema of the loaded DataFrame
# * There are no column names
# * All values are loaded as strings

# In[2]:


rawData


# #### View the values in the top 5 rows

# In[3]:


rawData.show(5)


# #### Assign names to each of the columns
# And create a new dataframe from it

# In[4]:


dataset = rawData.toDF('Label',
                'Alcohol',
                'MalicAcid',
                'Ash',
                'AshAlkalinity',
                'Magnesium',
                'TotalPhenols',
                'Flavanoids',
                'NonflavanoidPhenols',
                'Proanthocyanins',
                'ColorIntensity',
                'Hue',
                'OD',
                'Proline'
                )


# #### Confirm that the dataset contains the column names

# In[5]:


dataset


# #### View the dataset with the values

# In[6]:


dataset.show(5)


# #### Define a vectorize function to store the data in the required format for our ML models
# The ML package needs data be put in a (label: Double, features: Vector) DataFrame format with correspondingly named fields. The vectorize() function does just that
# * We perform a manual transformation of our dataset here
# * Spark ML also supplies built-in transformers which we will use shortly

# In[7]:


from pyspark.ml.linalg import Vectors

def vectorize(data):
    return data.rdd.map(lambda r: [r[0], Vectors.dense(r[1:])]).toDF(['label','features'])


# #### Convert our data set into the vectorized format

# In[8]:


vectorizedData = vectorize(dataset)


# In[9]:


vectorizedData.show(5)


# #### View the transformed dataset
# The features are now a DenseVector with an array of feature values

# In[10]:


vectorizedData.take(5)


# #### StringIndexer 
# * It's a feature transformer (can also be used for labels)
# * Encodes a string column to a column of indices. The indices are in [0, numLabels), ordered by value frequencies, so the most frequent value gets index 0
# * The label needs to be of type Double which will be handled by StringIndexer

# In[11]:


from pyspark.ml.feature import StringIndexer

labelIndexer = StringIndexer(inputCol='label',
                             outputCol='indexedLabel')


# #### Transform the label in the vectorized dataset with the StringIndexer
# We get a new label field called indexedLabel

# In[12]:


indexedData = labelIndexer.fit(vectorizedData).transform(vectorizedData)
indexedData.take(2)


# #### Confirm that the indexedLabel is in Double format

# In[13]:


indexedData


# In[14]:


indexedData.select('label').distinct().show()


# In[15]:


indexedData.select('indexedLabel').distinct().show()


# #### Split the vectorized data into training and test sets

# In[16]:


(trainingData, testData) = indexedData.randomSplit([0.8, 0.2])


# ### DecisionTree Classifier
# * Specify the features and label columns
# * <b>maxDepth: </b>The maximum depth of the decision tree
# * <b>impurity: </b>We use gini instead of entropy. Gini measurement is the probability of a random sample being classified correctly. Entropy is a measure of information (seek to maximize information gain when making a split). Outputs generally don't vary much when either option is chosen, but entropy may take longer to compute as it calculates a logarithm

# In[17]:


from pyspark.ml.classification import DecisionTreeClassifier

dtree = DecisionTreeClassifier(
    labelCol='indexedLabel', 
    featuresCol='features',
    maxDepth=3,
    impurity='gini'
)


# #### Traing the model using the training data

# In[18]:


model = dtree.fit(trainingData)


# #### Use Spark ML's MulticlassClassificationEvaluator to evaluate the model
# * Used to evaluate classification models
# * It takes a set of labels and predictions as input
# * Similar to (but not the same as MulticlassMetrics in MLLib)
# * <b>metricName: </b>Can be precision, recall, weightedPrecision, weightedRecall and f1

# In[19]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol='indexedLabel',
                                              predictionCol='prediction', 
                                              metricName='f1')


# #### Transform the test data using our model to include predictions

# In[24]:


transformed_data = model.transform(testData)
transformed_data.show(5)


# #### Measure accuracy of model on the test data

# In[25]:


print(evaluator.getMetricName(), 
      'accuracy:', 
      evaluator.evaluate(transformed_data))


# #### View only the columns relevant for the predictions

# In[22]:


predictions = transformed_data.select('indexedLabel', 'prediction', 'probability')
predictions.show(5)


# #### Spark dataframes can also be converted to Pandas dataframes
# View our predictions as a Pandas dataframe

# In[23]:


predictions.toPandas().head()

