
# coding: utf-8

# ### Analyzing airline data with Spark SQL

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName("Analyzing airline data")     .getOrCreate()


# In[74]:


from pyspark.sql.types import Row
from datetime import datetime


# #### Loading in airline data

# In[15]:


airlinesPath = "../datasets/airlines.csv"
flightsPath = "../datasets/flights.csv"
airportsPath = "../datasets/airports.csv"


# In[4]:


airlines = spark.read                .format("csv")                .option("header", "true")                .load(airlinesPath)


# In[5]:


airlines.createOrReplaceTempView("airlines")


# In[17]:


airlines = spark.sql("SELECT * FROM airlines")
airlines.columns


# In[18]:


airlines.show(5)


# In[19]:


flights = spark.read               .format("csv")               .option("header", "true")               .load(flightsPath)


# In[21]:


flights.createOrReplaceTempView("flights")

flights.columns


# In[22]:


flights.show(5)


# #### Counting with dataframes

# In[23]:


flights.count(), airlines.count()


# #### Counting using SQL

# In[25]:


flights_count = spark.sql("SELECT COUNT(*) FROM flights")
airlines_count = spark.sql("SELECT COUNT(*) FROM airlines")


# In[26]:


flights_count, airlines_count


# In[27]:


flights_count.collect()[0][0], airlines_count.collect()[0][0]


# #### Dataframes created using SQL commands can be aggregated, grouped etc. exactly as before

# In[28]:


total_distance_df = spark.sql("SELECT distance FROM flights")                         .agg({"distance":"sum"})                         .withColumnRenamed("sum(distance)","total_distance")


# In[29]:


total_distance_df.show()


# #### Analyzing flight delays

# In[51]:


all_delays_2012 = spark.sql(
    "SELECT date, airlines, flight_number, departure_delay " +
    "FROM flights WHERE departure_delay > 0 and year(date) = 2012")


# In[53]:


all_delays_2012.show(5)


# In[54]:


all_delays_2014 = spark.sql(
    "SELECT date, airlines, flight_number, departure_delay " +
    "FROM flights WHERE departure_delay > 0 and year(date) = 2014")

all_delays_2014.show(5)


# In[55]:


all_delays_2014.createOrReplaceTempView("all_delays")


# In[56]:


all_delays_2014.orderBy(all_delays_2014.departure_delay.desc()).show(5)


# #### Total number of delayed flights in 2014

# In[57]:


delay_count = spark.sql("SELECT COUNT(departure_delay) FROM all_delays")


# In[58]:


delay_count.show()


# In[59]:


delay_count.collect()[0][0]


# #### Percentage of flights delayed

# In[61]:


delay_percent = delay_count.collect()[0][0] / flights_count.collect()[0][0] * 100
delay_percent


# ### Finding delay per aIrlines

# In[62]:


delay_per_airline = spark.sql("SELECT airlines, departure_delay FROM flights")                         .groupBy("airlines")                         .agg({"departure_delay":"avg"})                         .withColumnRenamed("avg(departure_delay)", "departure_delay")


# In[63]:


delay_per_airline.orderBy(delay_per_airline.departure_delay.desc()).show(5)


# In[64]:


delay_per_airline.createOrReplaceTempView("delay_per_airline")


# In[65]:


delay_per_airline = spark.sql("SELECT * FROM delay_per_airline ORDER BY departure_delay DESC")


# In[66]:


delay_per_airline.show(5)


# #### SQL join operations 
# 
# * Get the names of the delayed flights

# In[70]:


delay_per_airline = spark.sql("SELECT * FROM delay_per_airline " +
                              "JOIN airlines ON airlines.code = delay_per_airline.airlines " +
                              "ORDER BY departure_delay DESC")

delay_per_airline.show(5)


# In[71]:


delay_per_airline.drop("code").show(5)

