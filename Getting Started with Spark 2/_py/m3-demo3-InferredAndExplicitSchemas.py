
# coding: utf-8

# ### Inferred and explicit schemas

# In[26]:


from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName("Inferred and explicit schemas")     .getOrCreate()


# In[46]:


from pyspark.sql.types import Row


# #### Inferring schema

# In[33]:


lines = sc.textFile("../datasets/students.txt")


# In[34]:


lines.collect()


# In[35]:


parts = lines.map(lambda l: l.split(","))

parts.collect()


# In[36]:


students = parts.map(lambda p: Row(name=p[0], math=int(p[1]), english=int(p[2]), science=int(p[3])))


# In[37]:


students.collect()


# In[38]:


schemaStudents = spark.createDataFrame(students)

schemaStudents.createOrReplaceTempView("students")


# In[39]:


schemaStudents.columns


# In[40]:


schemaStudents.schema


# In[43]:


spark.sql("SELECT * FROM students").show()


# #### Explicit schema

# In[44]:


parts.collect()


# In[45]:


schemaString = "name math english science"


# In[56]:


from pyspark.sql.types import StructType, StructField, StringType, LongType

fields = [StructField('name', StringType(), True),
          StructField('math', LongType(), True),
          StructField('english', LongType(), True),
          StructField('science', LongType(), True),
]


# In[57]:


schema = StructType(fields)


# In[58]:


schemaStudents = spark.createDataFrame(parts, schema)


# In[61]:


schemaStudents.columns


# In[60]:


schemaStudents.schema


# In[62]:


spark.sql("SELECT * FROM students").show()

