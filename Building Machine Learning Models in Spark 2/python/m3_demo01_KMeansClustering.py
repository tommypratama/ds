
# coding: utf-8

# ### Download the dataset
# <b>Dataset location: </b>https://www.kaggle.com/c/3136/download/train.csv

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName('Examine data about passengers on the Titanic')     .getOrCreate()

rawData = spark.read            .format('csv')            .option('header', 'true')            .load('../datasets/titanic.csv')


# In[2]:


rawData.toPandas().head()


# #### Select the columns which we required
# Also cast the numeric values as float

# In[3]:


from pyspark.sql.functions import col

dataset = rawData.select(col('Survived').cast('float'),
                         col('Pclass').cast('float'),
                         col('Sex'),
                         col('Age').cast('float'),
                         col('Fare').cast('float'),
                         col('Embarked')
                        )

dataset.toPandas().head()


# #### Drop rows containing missing values

# In[4]:


dataset = dataset.replace('?', None)        .dropna(how='any')


# #### Define StringIndexers for categorical columns

# In[5]:


from pyspark.ml.feature import StringIndexer

dataset = StringIndexer(
    inputCol='Sex', 
    outputCol='Gender', 
    handleInvalid='keep').fit(dataset).transform(dataset)

dataset = StringIndexer(
    inputCol='Embarked', 
    outputCol='Boarded', 
    handleInvalid='keep').fit(dataset).transform(dataset)

dataset.toPandas().head()


# #### Drop the redundant columns

# In[6]:


dataset = dataset.drop('Sex')
dataset = dataset.drop('Embarked')

dataset.toPandas().head()


# #### Define the required features to use in the VectorAssembler
# Since we are only examining data and not making predictions, we include all columns

# In[7]:


requiredFeatures = ['Survived',
                    'Pclass',
                    'Age',
                    'Fare',
                    'Gender',
                    'Boarded'
                   ]


# #### The VectorAssembler vectorises all the features
# The transformed data will be used for clustering

# In[8]:


from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=requiredFeatures, outputCol='features')


# #### Transorm our dataset for use in our clustering algorithm

# In[9]:


transformed_data = assembler.transform(dataset)


# In[10]:


transformed_data.toPandas().head()


# ### Define the clustering model
# Use K-means clustering
# * <b>k: </b>Defines the number of clusters
# * <b>seed: </b>This value is used to set the cluster centers. A different value of seed for the same k will result in clusters being defined differently. In order to reproduce similar clusters when re-running the clustering algorithm use the same values of k and seed

# In[11]:


from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=5, seed=3)
model = kmeans.fit(transformed_data)


# #### Create the clusters using the model

# In[12]:


clusterdData = model.transform(transformed_data)


# #### Use ClusteringEvaluator to evaluate the clusters
# <b>From Wikipedia: </b>The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.

# In[13]:


from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(clusterdData)
print('Silhouette with squared euclidean distance = ', silhouette)


# #### View the cluster centers for each of the features

# In[14]:


centers = model.clusterCenters()
print('Cluster Centers: ')
for center in centers:
    print(center)


# #### View the output of the KMeans model
# The prediction field denotes the cluster number

# In[15]:


clusterdData.toPandas().head()


# #### Get the average of each feature in the original data
# This is the equivalent of the cluster center when our dataset is one big cluster
# * We import all sql functions as we need the avg and count functions among others

# In[16]:


from pyspark.sql.functions import *

dataset.select(avg('Survived'),
               avg('Pclass'),
               avg('Age'),
               avg('Fare'),
               avg('Gender'),
               avg('Boarded')).toPandas()


# #### A more intuitive way to view the cluster centers in our clusterdData
# * We group by clusterID (prediction) and compute the average of all features
# * We do a count of values in each cluster

# In[17]:


clusterdData.groupBy('prediction').agg(avg('Survived'),
                                      avg('Pclass'),
                                      avg('Age'),
                                      avg('Fare'),
                                      avg('Gender'),
                                      avg('Boarded'),
                                      count('prediction')
                                     ).orderBy('prediction').toPandas()


# #### Examine all rows in one of the clusters

# In[18]:


clusterdData.filter(clusterdData.prediction == 1).toPandas()

