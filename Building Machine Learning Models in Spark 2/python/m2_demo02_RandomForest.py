
# coding: utf-8

# <b>Dataset location: </b>https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
# 
# The data is in the format value1, value2... <br />
# The leading whitespace for each value needs to be removed

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName('Predicting whether a person\'s income is greater than $50K')     .getOrCreate()

rawData = spark.read            .format('csv')            .option('header', 'false')            .option('ignoreLeadingWhiteSpace', 'true')            .load('../datasets/adult.csv')


# #### Specify column headers for data set

# In[2]:


dataset = rawData.toDF('Age',
               'WorkClass',
               'FnlWgt',
               'Education',
               'EducationNum',
               'MaritalStatus',
               'Occupation',
               'Relationship',
               'Race',
               'Gender',
               'CapitalGain',
               'CapitalLoss',
               'HoursPerWeek',
               'NativeCountry',
               'Label'
                )


# In[3]:


dataset.toPandas().head()


# #### Drop FnlWgt column which does not appear meaningful

# In[4]:


dataset = dataset.drop('FnlWgt')


# #### Examine the dataset
# * The FnlWgt column has been dropped
# * There are missing values in the data represented by '?' (e.g. line 32541 for column WorkClass)

# In[5]:


dataset.toPandas()


# #### Count rows in dataset

# In[6]:


dataset.count()


# #### Convert missing values to null
# Missing values in this dataset are represented by ?

# In[7]:


dataset = dataset.replace('?', None)


# #### Drop all rows which contain even a single missing value
# The value 'any' for parameter how specifies that even a single missing value in a row should result in it being dropped (as opposed to 'all' where all values need to be missing)

# In[8]:


dataset = dataset.dropna(how='any')


# #### Number of rows has reduced now

# In[9]:


dataset.count()


# #### Confirm missing value rows are not there
# Row 32541 for example

# In[10]:


dataset.toPandas()


# #### View the data types for all the columns
# Since they have all been loaded as Strings, we need to convert the numeric fields to Float

# In[11]:


dataset.describe()


# In[12]:


from pyspark.sql.types import FloatType
from pyspark.sql.functions import col

dataset = dataset.withColumn('Age', 
                             dataset['Age'].cast(FloatType()))
dataset = dataset.withColumn('EducationNum', 
                             dataset['EducationNum'].cast(FloatType()))
dataset = dataset.withColumn('CapitalGain', 
                             dataset['CapitalGain'].cast(FloatType()))
dataset = dataset.withColumn('CapitalLoss', 
                             dataset['CapitalLoss'].cast(FloatType()))
dataset = dataset.withColumn('HoursPerWeek', 
                             dataset['HoursPerWeek'].cast(FloatType()))

dataset.toPandas().head()


# #### Transform categorical fields
# First use StringIndexer to convert categorical values to indices

# In[13]:


from pyspark.ml.feature import StringIndexer

indexedDF = StringIndexer(
    inputCol='WorkClass', 
    outputCol='WorkClass_index').fit(dataset).transform(dataset)


# #### A new column called WorkClass_index is created
# This stores the indexed values of WorkClass

# In[14]:


indexedDF.toPandas().head()


# #### OneHotEncoding
# Use the new indexed field to obtain a one-hot-encoded field

# In[15]:


from pyspark.ml.feature import OneHotEncoder

encodedDF = OneHotEncoder(
    inputCol="WorkClass_index", 
    outputCol="WorkClass_encoded").transform(indexedDF)


# #### A WorkClass_encoded field is created 
# * This contains the one-hot-encoding for WorkClass
# * This cannot operate directly on a column with string values - values need to be numeric. Hence we use the WorkClass_index as input

# In[16]:


encodedDF.toPandas().head()


# #### View the original and transformed fields together

# In[17]:


encodedDF.select('WorkClass', 'WorkClass_index', 'WorkClass_encoded')         .toPandas()         .head()


# ### Transform the entire dataset
# * So far we have only transformed a single column
# * We need to perform this transformation for every categorical and non-numeric column
# * This will be simplified by using a Pipeline (a feature of Spark ML)

# ####  First, split the data into training and test sets

# In[18]:


(trainingData, testData) = dataset.randomSplit([0.8,0.2])


# #### Encode all the categorical fields in the dataset
# We begin by listing all the categorical fields

# In[19]:


categoricalFeatures = [
               'WorkClass',
               'Education',
               'MaritalStatus',
               'Occupation',
               'Relationship',
               'Race',
               'Gender',
               'NativeCountry'
]


# #### Create an array of StringIndexers to convert the categorical values to indices

# In[20]:


indexers = [StringIndexer(
    inputCol=column, 
    outputCol=column + '_index', 
    handleInvalid='keep') for column in categoricalFeatures]


# #### Create an array of OneHotEncoders to encode the categorical values

# In[21]:


encoders = [OneHotEncoder(
    inputCol=column + '_index', 
    outputCol= column + '_encoded') for column in categoricalFeatures]


# #### Index the Label field

# In[22]:


labelIndexer = [StringIndexer(
    inputCol='Label', outputCol='Label_index')]


# #### Create a pipeline
# The pipeline contains the array of StringIndexers and OneHotEncoders

# In[23]:


from pyspark.ml import Pipeline

pipeline = Pipeline(stages=indexers + encoders + labelIndexer)


# #### View the result of the transformations performed by this pipeline
# This pipeline can transform our dataset into a format which can be used by our model

# In[24]:


transformedDF = pipeline.fit(trainingData).transform(trainingData)
transformedDF.toPandas().tail()


# #### Select the required features
# At this point the dataset contains a lot of additional columns. We select the features needed by our model

# In[25]:


requiredFeatures = [
    'Age',
    'EducationNum',
    'CapitalGain',
    'CapitalLoss',
    'HoursPerWeek',
    'WorkClass_encoded',
    'Education_encoded',
    'MaritalStatus_encoded',
    'Occupation_encoded',
    'Relationship_encoded',
    'Race_encoded',
    'Gender_encoded',
    'NativeCountry_encoded'
]


# #### VectorAssembler
# VectorAssembler is a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector
# * We had previously written our own function to create such a vector

# In[26]:


from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=requiredFeatures, outputCol='features')


# In[27]:


transformedDF = assembler.transform(transformedDF)
transformedDF.toPandas().tail()


# In[28]:


transformedDF.select('features').toPandas().tail()


# #### Specify our estimator

# In[29]:


from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol='Label_index', 
                            featuresCol='features',
                            maxDepth=5)


# #### Final Pipeline
# * The pipeline we built previously only transformed the feature columns
# * We re-create the pipeline to include the VectorAssembler and the estimator
# 
# The pipeline to be used to build the model contains all the transformers and ends with the estimator

# In[30]:


pipeline = Pipeline(
    stages=indexers + encoders + labelIndexer + [assembler, rf]
)


# #### Train the model

# In[31]:


model = pipeline.fit(trainingData)


# #### Use the test data for predictions

# In[32]:


predictions = model.transform(testData)
predictionsDF = predictions.toPandas()
predictionsDF.head()


# #### Select the correct label and predictions to evaluate the model

# In[34]:


predictions = predictions.select(
    'Label_index',
    'prediction'
)


# #### Create an evaluator for our model

# In[35]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol='Label_index', 
    predictionCol='prediction', 
    metricName='accuracy')


# #### Check the accuracy

# In[36]:


accuracy = evaluator.evaluate(predictions)
print('Test Accuracy = ', accuracy)


# #### Examine incorrect predictions

# In[37]:


predictionsDF.loc[
    predictionsDF['Label_index'] != predictionsDF['prediction']
]

