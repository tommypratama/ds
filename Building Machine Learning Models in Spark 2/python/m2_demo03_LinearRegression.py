
# coding: utf-8

# ### Download the Automobile data set
# <b>Download link:</b> https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName('Predicting the price of an automobile given a set of features')     .getOrCreate()

rawData = spark.read            .format('csv')            .option('header', 'true')            .load('../datasets/imports-85.data')


# In[2]:


rawData.toPandas().head()


# #### Select the required columns
# * We can select the specific features we feel are relevant in our dataset
# * fields such as normalized-losses have been dropped
# * The numeric fields can be cast as float or any numeric type

# In[3]:


from pyspark.sql.functions import col

dataset = rawData.select(col('price').cast('float'), 
                         col('make'), 
                         col('num-of-doors'), 
                         col('body-style'), 
                         col('drive-wheels'), 
                         col('wheel-base').cast('float'), 
                         col('curb-weight').cast('float'), 
                         col('num-of-cylinders'), 
                         col('engine-size').cast('float'), 
                         col('horsepower').cast('float'), 
                         col('peak-rpm').cast('float')
                        )


# In[4]:


dataset.toPandas().head()


# #### Drop columns with nulls
# Check number of rows in dataset before and after removal of nulls

# In[5]:


dataset.count()


# In[6]:


dataset = dataset.replace('?', None).dropna(how='any')


# In[7]:


dataset.count()


# #### Split dataset into training and test sets

# In[8]:


(trainingData, testData) = dataset.randomSplit([0.8,0.2])


# #### List the categorical fields so that we can transform these to encoded values

# In[9]:


categoricalFeatures = ['make',
                       'num-of-doors',
                       'body-style',
                       'drive-wheels',
                       'num-of-cylinders'
                      ]                     


# #### Import and implement the required transformers

# In[10]:


from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler


# #### Use of handleInvalid in StringIndexer
# If the model comes across a new label which it hasn't seen in the training phase, it is deemed an "invalid" label. There are different ways of handling this:
# * handleInvalid='skip' will remove rows with new labels
# * handleInvalid='error' will cause an error when a new label is encountered
# * handleInvalid='keep' will create a new index if it encounters a new label (available from Spark 2.2 onwards)

# In[11]:


indexers = [StringIndexer(
    inputCol=column, 
    outputCol=column + '_index', 
    handleInvalid='keep') for column in categoricalFeatures]


# #### One-Hot-Encode the features

# In[12]:


encoders = [OneHotEncoder(
    inputCol=column + '_index', 
    outputCol= column + '_encoded') for column in categoricalFeatures]


# #### List all the required features from the transformed dataset

# In[13]:


requiredFeatures = ['make_encoded',
                    'num-of-doors_encoded',
                    'body-style_encoded',
                    'drive-wheels_encoded',
                    'wheel-base',
                    'curb-weight',
                    'num-of-cylinders_encoded',
                    'engine-size',
                    'horsepower',
                    'peak-rpm'
                   ]


# #### Prepare the feature assembler

# In[14]:


assembler = VectorAssembler(inputCols=requiredFeatures, outputCol='features')


# #### Linear Regression
# 
# 
# 
# By setting α properly, elastic net contains both L1 and L2 regularization as special cases. 
# * If the elasticNetParam α is set to 1, it is equivalent to a Lasso model
# * If α is set to 0, the trained model reduces to a ridge regression model
# 
# regParam is the regularization variable

# In[15]:


from pyspark.ml.regression import LinearRegression

lr = LinearRegression(maxIter=100,
                      regParam=1.0,
                      elasticNetParam=0.8,
                      labelCol='price', 
                      featuresCol='features')


# #### Define our pipeline
# It contains all our transformers plus the model

# In[16]:


from pyspark.ml import Pipeline

pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])


# In[17]:


model = pipeline.fit(trainingData)
model


# ### Extract the model from the pipeline
# * Our LinearRegression model is not the same as the pipeline model
# * To extract the LinearRegression model, we get it from the last stage of our pipeline model

# In[18]:


lrModel = model.stages[-1]


# In[19]:


print('Training RMSE = ', lrModel.summary.rootMeanSquaredError)
print('Training R^2 score = ', lrModel.summary.r2)


# #### Check the number of features
# The number will be high as many of our features are one-hot-encoded

# In[20]:


lrModel.numFeatures


# #### View the coefficients of each feature

# In[21]:


lrModel.coefficients


# #### There is a coefficient for each feature

# In[22]:


len(lrModel.coefficients)


# #### Get predictions using our model on the test data

# In[23]:


predictions = model.transform(testData)
predictionsDF = predictions.toPandas()
predictionsDF.head()


# #### The features have been transformed to LibSVM format

# In[24]:


predictionsDF['features'][0]


# ### Use RegressionEvaluator to evaluate the model
# * MulticlassClassificationEvaluator is used for classification models
# * RegressionEvaluator needed to evaluate regression models
# * <b>metricName </b>can be r2, rmse, mse or mae (mean absolute error)

# In[25]:


from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    labelCol='price', 
    predictionCol='prediction', 
    metricName='r2')

r2 = evaluator.evaluate(predictions)
print('Test R^2 score = ', r2)


# In[26]:


evaluator = RegressionEvaluator(
    labelCol='price', 
    predictionCol='prediction', 
    metricName='rmse')

rmse = evaluator.evaluate(predictions)
print('Test RMSE = ', rmse)


# #### Compare the actual and predicted values of price

# In[27]:


predictionsPandasDF = predictions.select(
    col('price'),
    col('prediction')
).toPandas()


# In[28]:


predictionsPandasDF.head()


# #### Plot a graph of actual and predicted values of price
# Note that our predictions dataset is sorted in ascending order of price

# In[29]:


import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))

plt.plot(predictionsPandasDF['price'], label='Actual')
plt.plot(predictionsPandasDF['prediction'], label='Predicted')

plt.ylabel('Price')
plt.legend()

plt.show()


# ### Using ParamGrid for hyperparameter tuning
# The parameters we wish to tweak are:
# * maxIter
# * regParam
# * elasticNetParam - whether a lasso or ridge model will be best

# In[30]:


from pyspark.ml.tuning import ParamGridBuilder

paramGrid = ParamGridBuilder().addGrid(
    lr.maxIter, [10,50,100]).addGrid(
    lr.regParam, [0.1, 0.3, 1.0]).addGrid(
    lr.elasticNetParam, [0.0, 1.0]).build()


# #### Define the RegressionEvaluator used to evaluate the models
# We wish to minimize RMSE

# In[31]:


evaluator = RegressionEvaluator(
    labelCol='price', 
    predictionCol='prediction', 
    metricName='rmse')


# ### Define the CrossValidator
# This is used to put all the pieces together
# * <b>estimator: </b>Can be a standalone estimator or a pipeline with an estimator at the end. We use our pipeline
# * <b>estimatorParamMaps: </b>We add our paramGrid in order to build models with different combinations of the parameters
# * <b>evaluator: </b>To evaluate each model, we specify our evaluator

# In[32]:


from pyspark.ml.tuning import CrossValidator

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)


# #### Train each of our models with the training data
# After identifying the best ParamMap, CrossValidator finally re-fits the Estimator using the best ParamMap and the entire dataset

# In[33]:


model = crossval.fit(trainingData)


# #### To examine our best model, we extract it from the pipeline

# In[34]:


lrModel = model.bestModel.stages[-1]
lrModel


# #### Get the values of the "best" parameters
# Unfortunately, extracting these values is a bit awkward as we need to access the \_java\_obj object 

# In[35]:


print('maxIter=', lrModel._java_obj.getMaxIter())
print('elasticNetParam=', lrModel._java_obj.getElasticNetParam())
print('regParam=', lrModel._java_obj.getRegParam())


# #### Make predictions using our "best" model

# In[36]:


predictions = model.transform(testData)
predictionsDF = predictions.toPandas()
predictionsDF.head()


# #### Evaluate the model on it's R-square score and RMSE

# In[37]:


evaluator = RegressionEvaluator(
    labelCol='price', 
    predictionCol='prediction', 
    metricName='r2')

rsquare = evaluator.evaluate(predictions)
print("Test R^2 score = %g" % rsquare)


# In[38]:


evaluator = RegressionEvaluator(
    labelCol='price', 
    predictionCol='prediction', 
    metricName='rmse')

rmse = evaluator.evaluate(predictions)
print('Test RMSE = ', rmse)


# #### Compare actual and predicted values of price

# In[39]:


predictionsPandasDF = predictions.select(
    col('price'),
    col('prediction')).toPandas()

predictionsPandasDF.head()


# #### Perform the comparison using a graph

# In[40]:


plt.figure(figsize=(15,6))

plt.plot(predictionsPandasDF['price'], label='Actual')
plt.plot(predictionsPandasDF['prediction'], label='Predicted')

plt.ylabel('Price')
plt.legend()

plt.show()

