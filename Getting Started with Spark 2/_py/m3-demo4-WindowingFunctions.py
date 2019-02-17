
# coding: utf-8

# ### Window functions

# In[2]:


from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName("Window functions")     .getOrCreate()


# In[3]:


products = spark.read                .format("csv")                .option("header", "true")                .load('../datasets/products.csv')


# In[4]:


products.show()


# #### Window rank function

# In[5]:


import sys
from pyspark.sql.window import Window
import pyspark.sql.functions as func


# In[6]:


windowSpec1 = Window     .partitionBy(products['category'])     .orderBy(products['price'].desc())


# In[7]:


price_rank = (func.rank().over(windowSpec1))


# In[8]:


product_rank = products.select(
        products['product'],
        products['category'],
        products['price']
).withColumn('rank', func.rank().over(windowSpec1))

product_rank.show()


# #### Window max function between rows

# In[9]:


windowSpec2 = Window     .partitionBy(products['category'])     .orderBy(products['price'].desc())     .rowsBetween(-1, 0)


# In[10]:


price_max = (func.max(products['price']).over(windowSpec2))


# In[11]:


products.select(
    products['product'],
    products['category'],
    products['price'],
    price_max.alias("price_max")).show()


# #### Window price difference function between ranges

# In[12]:


windowSpec3 = Window     .partitionBy(products['category'])     .orderBy(products['price'].desc())     .rangeBetween(-sys.maxsize, sys.maxsize)


# In[13]:


price_difference =   (func.max(products['price']).over(windowSpec3) - products['price'])


# In[14]:


products.select(
    products['product'],
    products['category'],
    products['price'],
    price_difference.alias("price_difference")).show()


# In[104]:


windowSpec4 = Window     .partitionBy(products['category'])     .orderBy(products['price'].asc())     .rangeBetween(0, sys.maxsize)


# In[105]:


sys.maxsize


# In[106]:


price_max = (func.max(products['price']).over(windowSpec4))


# In[107]:


products.select(
    products['product'],
    products['category'],
    products['price'],
    price_max.alias("price_max")).show()

