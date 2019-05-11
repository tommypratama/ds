
# coding: utf-8

# ### Download the dataset
# <b>Dataset location: </b>https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset <br />
# Number of riders using a bikeshare service on a given day. We will predict the number of riders given information about the type of day and weather

# In[3]:


from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName('Effects of Dimensionality Reduction when making predictions')     .getOrCreate()

rawData = spark.read            .format('csv')            .option('header', 'true')            .load('../datasets/day.csv')


# In[4]:


rawData.toPandas().head()


# #### Select required columns from data
# * The <i>instant</i> and <i>dteday</i> columns are dropped as they are unique, thus not useful for predictions
# * The <i>casual</i> and <i>registered</i> fields will sum up to the cnt field which we wish to predict, so we remove those

# In[4]:


from pyspark.sql.functions import col

dataset = rawData.select(col('season').cast('float'),
                         col('yr').cast('float'),
                         col('mnth').cast('float'),
                         col('holiday').cast('float'),
                         col('weekday').cast('float'),
                         col('workingday').cast('float'),
                         col('weathersit').cast('float'),
                         col('temp').cast('float'),
                         col('atemp').cast('float'),
                         col('hum').cast('float'),
                         col('windspeed').cast('float'),
                         col('cnt').cast('float')
                        )

dataset.toPandas().head()


# #### Check correlation between fields
# * <i>temp</i> and <i>atemp</i> are almost perfectly correlated
# * <i>month</i> and <i>season</i> are also strongly correlated

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

corrmat = dataset.toPandas().corr()
plt.figure(figsize=(8, 8))
sns.set(font_scale=1.0)
sns.heatmap(corrmat, vmax=.8, square=True, annot=True, fmt='.2f', cmap = "winter")
plt.show()


# #### Select features
# All fields except the <i>cnt</i> field make up our features

# In[6]:


featureCols = dataset.columns.copy()
featureCols.remove('cnt')

featureCols


# ### The total number of features
# We will reduce the dimensions later on

# In[7]:


len(featureCols)


# #### Construct the feature vector using an assembler

# In[8]:


from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=featureCols,
                            outputCol='features')


# In[9]:


vectorDF = assembler.transform(dataset)
vectorDF.toPandas().head()


# #### Prepare the training and test data sets

# In[10]:


(trainingData, testData) = vectorDF.randomSplit([0.8, 0.2])


# #### Create a simple Linear Regression model
# We will not aim to get the best parameters here as our aim is to compare the model when used with regular features and then features with transformed and reduced dimensions

# In[17]:


from pyspark.ml.regression import LinearRegression

lr = LinearRegression(maxIter=100,
                      regParam=1.0,
                      elasticNetParam=0.8,
                      labelCol='cnt', 
                      featuresCol='features')


# In[18]:


model = lr.fit(trainingData)


# #### Calculate R-square and RMSE on training data

# In[19]:


print('Training R^2 score = ', model.summary.r2)
print('Training RMSE = ', model.summary.rootMeanSquaredError)


# #### Make predictions using test data

# In[20]:


predictions = model.transform(testData)
predictions.toPandas().head()


# #### R-square score on test data

# In[21]:


from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    labelCol='cnt', 
    predictionCol='prediction', 
    metricName='r2')

rsquare = evaluator.evaluate(predictions)
print("Test R^2 score = %g" % rsquare)


# #### RMSE on test data

# In[22]:


evaluator = RegressionEvaluator(
    labelCol='cnt', 
    predictionCol='prediction', 
    metricName='rmse')

rmse = evaluator.evaluate(predictions)
print('Test RMSE = ', rmse)


# #### Convert predictions dataframe to Pandas dataframe
# This will make it easier for us to create a series which we will use to plot the actual and predicted values of <i>cnt</i>

# In[73]:


predictionsPandas = predictions.toPandas()


# #### Compare actual and predicted values of cnt

# In[74]:


plt.figure(figsize=(15,6))

plt.plot(predictionsPandas['cnt'], label='Actual')
plt.plot(predictionsPandas['prediction'], label='Predicted')

plt.ylabel('Ride Count')
plt.legend()

plt.show()


# ### Principal Components Analysis
# Performs an orthogonal transformation to convert a set of possibly correlated variables into a set of values of linearly uncorrelated variables called <b>principal components</b>
# * the pcaTransformer will extract the principal components from the features
# * the number of components is set by the value of <b>k</b>

# In[23]:


from pyspark.ml.feature import PCA

pca = PCA(k=8, 
          inputCol='features', 
          outputCol='pcaFeatures'
         )


# In[24]:


pcaTransformer = pca.fit(vectorDF)


# #### View the principal components in the transformed space

# In[25]:


pcaFeatureData = pcaTransformer.transform(vectorDF).                select('pcaFeatures')

pcaFeatureData.toPandas().head()


# #### The principal components are stored as a DenseVector

# In[26]:


pcaFeatureData.toPandas()['pcaFeatures'][0]


# #### Check the  Explained Variance 
# This shows the eigen values of each principal component in decreasing order

# In[27]:


pcaTransformer.explainedVariance


# #### Scree plot
# Visualise the explained variance. This can be used to eliminate dimensions with low eigen values

# In[28]:


plt.figure(figsize=(15,6))

plt.plot(pcaTransformer.explainedVariance)
plt.xlabel('Dimension')
plt.ylabel('Explain Variance')
plt.show()


# #### Prepare a dataset to use with the Linear Regression model
# * We need to prepare a dataframe containing the principal components and the correct value of cnt
# * We will retrieve the <i>pcaFeatures</i> column from pcaFeatureData and the <i>cnt</i> column from vectorDF
# * In order to join these two dataframes, we need to create a column on which to join - for that we add a <i>row_index</i> field to each dataframe which we will use to perform the join

# In[29]:


from pyspark.sql.functions import monotonically_increasing_id

pcaFeatureData = pcaFeatureData.withColumn('row_index', monotonically_increasing_id())
vectorDF = vectorDF.withColumn('row_index', monotonically_increasing_id())


# #### Join the tables using the row_index field
# We only extract the <i>cnt</i> and <i>pcaFeatures</i> fields which we require from the joined table

# In[30]:


transformedData = pcaFeatureData.join(vectorDF, on=['row_index']).                sort('row_index').                select('cnt', 'pcaFeatures') 
        
transformedData.toPandas().head()


# #### Prepare the training and test datasets from our new transformed dataset

# In[31]:


(pcaTrainingData, pcaTestData) = transformedData.randomSplit([0.8,0.2])


# #### Prepare the LinearRegression model
# This has the exact same parameters as our previous model for a meaningful comparison

# In[32]:


pcalr = LinearRegression(maxIter=100,
                      regParam=1.0,
                      elasticNetParam=0.8,
                      labelCol='cnt', 
                      featuresCol='pcaFeatures')


# In[33]:


pcaModel = pcalr.fit(pcaTrainingData)


# #### Calculate RMSE and R-square values on training data

# In[34]:


print('Training R^2 score = ', pcaModel.summary.r2)
print('Training RMSE = ', pcaModel.summary.rootMeanSquaredError)


# #### Perform predictions using the principal components

# In[35]:


pcaPredictions = pcaModel.transform(pcaTestData)
pcaPredictions.toPandas().head()


# #### Calculate R-square on test

# In[36]:


evaluator = RegressionEvaluator(
    labelCol='cnt', 
    predictionCol='prediction', 
    metricName='r2')

rsquare = evaluator.evaluate(pcaPredictions)
print("Test R^2 score = %g" % rsquare)


# #### RMSE on test

# In[37]:


evaluator = RegressionEvaluator(
    labelCol='cnt', 
    predictionCol='prediction', 
    metricName='rmse')

rmse = evaluator.evaluate(pcaPredictions)
print('Test RMSE = ', rmse)


# #### Convert predictions dataframe to a pandas dataframe
# This will allow us to plot a graph of the predicted values against the actual values

# In[38]:


pcaPredictionsPandas = pcaPredictions.toPandas()


# In[39]:


plt.figure(figsize=(15,6))

plt.plot(pcaPredictionsPandas['cnt'], label='Actual')
plt.plot(pcaPredictionsPandas['prediction'], label='Predicted')

plt.ylabel('Ride Count')
plt.legend()

plt.show()

