
# coding: utf-8

# ### Download dataset
# <b>Dataset location: </b>http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip <br /><br />
# Given the number of times users have listened to songs of an artist, make artist recommendations for the user

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName('Use Implicit Collaborative Filtering for band recommendations')     .getOrCreate()

rawData = spark.read               .format('csv')               .option('delimiter', '\t')               .option('header', 'true')               .load('../datasets/lastfm/user_artists.dat')
                
rawData.toPandas().head()


# #### Extract all the columns and cast the values as int

# In[2]:


from pyspark.sql.functions import col

dataset = rawData.select(col('userID').cast('int'), 
                         col('artistID').cast('int'), 
                         col('weight').cast('int')
                        )

dataset


# #### Examine the weight field
# This lists the number of times the user has listened to songs of that artist

# In[3]:


dataset.select('weight').describe().toPandas()


# #### Standardize the weight column
# * In order to get recommendations using implicit feedback (such as number of times an artist has been listened to), we need to standardize the weight column
# * Pyspark does not contain a built-in standardizer for scalar data (only for vectors) which is why we standardize the column values on our own

# In[4]:


from pyspark.sql.functions import stddev, mean, col

df = dataset.select(mean('weight').alias('mean_weight'), 
                    stddev('weight').alias('stddev_weight'))\
            .crossJoin(dataset)\
            .withColumn('weight_scaled' , 
                        (col('weight') - col('mean_weight')) / col('stddev_weight'))
        
df.toPandas().head()


# #### Split the dataset into training and test sets

# In[5]:


(trainingData, testData) = df.randomSplit([0.8, 0.2])


# #### Define the ALS model
# The metrics used to evaluate ALS models which use implicit feedback are:
# * Mean Average Precision (MAP)
# * Normalized Discounted Cumulative Gain (NDCG)
# 
# These are not part of Pyspark yet so will need to be implemented by us (not covered in this course)

# In[6]:


from pyspark.ml.recommendation import ALS

als = ALS(maxIter=10, 
          regParam=0.1, 
          userCol='userID', 
          itemCol='artistID',
          implicitPrefs=True,
          ratingCol='weight_scaled',
          coldStartStrategy='drop')

model = als.fit(trainingData)


# #### Get the predictions from our model on the test data

# In[7]:


predictions = model.transform(testData)
predictions.toPandas().head()


# #### Examine the distribution of the original weights and the predictions

# In[8]:


predictionsPandas = predictions.select('weight_scaled', 'prediction').toPandas()
predictionsPandas.describe()


# #### Load the Artist information from the artists.dat file
# This will be used to map the artistID listed in the recommendation with the actual artist name

# In[9]:


artistData = spark.read                  .format('csv')                  .option('delimiter', '\t')                  .option('header', 'true')                  .load('../datasets/lastfm/artists.dat')
                
artistData.toPandas().head()


# #### Define a function to get the artist recommendations
# * Similar to what was done in the last exercise for movie recommendations
# * Note how the joining of the artistData and artistsDF is a little different - the ids have different name in each table (artistID vs id)

# In[10]:


from pyspark.sql.types import IntegerType

def getRecommendationsForUser(userId, numRecs):
    
    usersDF = spark.    createDataFrame([userId], IntegerType()).    toDF('userId')
    
    userRecs = model.recommendForUserSubset(usersDF, numRecs)
    
    artistsList = userRecs.collect()[0].recommendations
    artistsDF = spark.createDataFrame(artistsList)
    
    recommendedArtists = artistData    .join(artistsDF, 
          artistData.id == artistsDF.artistID)\
    .orderBy('rating', ascending=False)\
    .select('name', 'url', 'rating')
    
    return recommendedArtists


# #### Get full recommendations for a particular userID
# * Users 939, 2013 gets recommended a lot of rock and metal bands 
# * User 2 gets recommended 80s/90s European pop bands
# * Metal bands for user 107
# * 1726 gets pop recommendations

# In[34]:


getRecommendationsForUser(1726, 10).toPandas()


# In[35]:


userArtistRaw = dataset.filter(dataset.userID == 1726)

userArtistsInfo = artistData.join(userArtistRaw, 
                                  artistData.id==userArtistRaw.artistID)\
                            .orderBy('weight', ascending=False)\
                            .select('name', 'weight')

userArtistsInfo.toPandas()

