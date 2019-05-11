
# coding: utf-8

# ### Download dataset
# <b>Dataset location: </b>http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName('Use Collaborative Filtering for movie recommendations')     .getOrCreate()

rawData = spark.read            .format('csv')            .option('header', 'true')            .load('../datasets/movielens/ratings.csv')


# In[2]:


rawData.toPandas().head()


# #### Pick all columns except the timestamp

# In[3]:


from pyspark.sql.functions import col

dataset = rawData.select(col('userId').cast('int'), 
                         col('movieId').cast('int'), 
                         col('rating').cast('float')
                        )

dataset.toPandas().head()


# #### Check the distribution of rating in the dataset

# In[6]:


dataset.select('rating').toPandas().describe()


# #### Split into training and test data sets

# In[7]:


(trainingData, testData) = dataset.randomSplit([0.8, 0.2])


# ### Define the Collaborative Filtering model
# Uses the Alternating Least Squares algorithm to learn the latent factors
# * <b>maxIter: </b>The maximum number of iterations to run
# * <b>regParam: </b>Specifies the regularization parameter in ALS (defaults to 1.0)
# * <b>coldStartStrategy: </b> Strategy for handling unknown or new users/items during prediction (which was not encountered in training). Options are 'drop' and 'nan'. We will drop unknown users/items from the predictions

# In[8]:


from pyspark.ml.recommendation import ALS

als = ALS(maxIter=5, 
          regParam=0.1, 
          userCol='userId', 
          itemCol='movieId', 
          ratingCol='rating',
          coldStartStrategy='drop')


# #### Build the ALSModel using the model definition and training data

# In[9]:


model = als.fit(trainingData)


# #### Get the predictions for the test data

# In[10]:


predictions = model.transform(testData)
predictions.toPandas().head()


# #### Compare the distribution of values for ratings and predictions

# In[11]:


predictions.select('rating', 'prediction').toPandas().describe()


# #### Get the Root Mean Square Error on the test data

# In[12]:


from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(metricName='rmse', 
                                labelCol='rating',
                                predictionCol='prediction')

rmse = evaluator.evaluate(predictions)
print('RMSE = ', rmse)


# #### The ALS model can be used to get predictions for all users
# Specify the number of predictions you would like for each user

# In[13]:


userRecsAll = model.recommendForAllUsers(3)
userRecsAll


# #### View the recommendations
# For each userId there is a list of tuples representing a movieId and it's rating for the user

# In[15]:


userRecsAll.toPandas().head()


# #### Get the top user recommendations for each movie
# * The users who are most likely to like a particular movie
# * Get the top 3 users

# In[19]:


movieRecsAll = model.recommendForAllItems(3)
movieRecsAll.toPandas().head()


# #### Get recommendations for a subset of users
# * Start off by creating a list of users who make up our subset
# * Convert that list to a dataframe which will be used shortly

# In[16]:


from pyspark.sql.types import IntegerType

usersList = [148, 463, 267]
usersDF = spark.createDataFrame(usersList, IntegerType()).toDF('userId')

usersDF.take(3)


# #### Use the recommendForUserSubset function
# This gets the recommendations for specific users

# In[17]:


userRecs = model.recommendForUserSubset(usersDF, 5)
userRecs.toPandas()


# #### Extract recommendations for specific user
# * We get a list comprising a Row object which in turn contains a list of Rows
# * To get the movie names from the movieIds so we will need to perform some transformations

# In[20]:


userMoviesList = userRecs.filter(userRecs.userId == 148).select('recommendations')

userMoviesList.collect()


# #### Extract the list of recommendations
# We get the list of Rows contining the movieId and rating for the user

# In[21]:


moviesList = userMoviesList.collect()[0].recommendations
moviesList


# #### Create a DataFrame containing the movieId and rating as columns
# Use the moviesList created previously

# In[22]:


moviesDF = spark.createDataFrame(moviesList)
moviesDF.toPandas()


# #### The movie names are stored in a csv file called movies.csv
# Load that into another dataframe

# In[23]:


movieData = sqlContext.read.csv('../datasets/movielens/movies.csv',
                              header=True,
                              ignoreLeadingWhiteSpace= True)
movieData.toPandas().head()


# In[24]:


recommendedMovies = movieData.join(moviesDF, on=['movieId']).orderBy('rating', ascending=False).select('title', 'genres', 'rating')

recommendedMovies.toPandas()


# In[25]:


from pyspark.sql.types import IntegerType

def getRecommendationsForUser(userId, numRecs):
    
    usersDF = spark.    createDataFrame([userId], IntegerType()).    toDF('userId')
    
    userRecs = model.recommendForUserSubset(usersDF, numRecs)
    
    moviesList = userRecs.collect()[0].recommendations
    moviesDF = spark.createDataFrame(moviesList)
    
    recommendedMovies = movieData.join(moviesDF, on=['movieId'])    .orderBy('rating', ascending=False)    .select('title', 'genres', 'rating')
    
    return recommendedMovies


# In[27]:


recommendationsForUser = getRecommendationsForUser(219, 10)
recommendationsForUser.toPandas()

