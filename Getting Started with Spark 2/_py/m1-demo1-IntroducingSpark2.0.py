
# coding: utf-8

# ## RDDs and DataFrames
# 
# * Creating RDDs and DataFrames using SparkContext
# * Interoperability between RDDs and DataFrames
# * Multiple rows and multiple column specifications for DataFrames
# * Creating DataFrames using SQLContext
# * Selecting, editing and renaming columns in dataframes
# * Interoperability between Pandas and Spark dataframes

# In[1]:


sc


# In[65]:


from pyspark.sql.types import Row
from datetime import datetime


# #### Creating RDDs using sc.parallelize()

# In[28]:


simple_data = sc.parallelize([1, "Alice", 50])
simple_data


# In[29]:


simple_data.count()


# In[30]:


simple_data.first()


# In[31]:


simple_data.take(2)


# In[32]:


simple_data.collect()


# #### This is an ERROR!
# 
# * This RDD does not have "columns", it cannot be represented as a tabular data frame
# * DataFrames are structured datasets

# In[ ]:


df = simple_data.toDF()


# #### RDDs with records using sc.parallelize()

# In[42]:


records = sc.parallelize([[1, "Alice", 50], [2, "Bob", 80]])
records


# In[43]:


records.collect()


# In[44]:


records.count()


# In[45]:


records.first()


# In[46]:


records.take(2)


# In[47]:


records.collect()


# #### This is an NOT an error!
# 
# * This RDD does have "columns", it can be represented as a tabular data frame

# In[48]:


df = records.toDF()


# In[49]:


df


# In[50]:


df.show()


# #### Creating dataframes using sc.parallelize() and Row() functions
# * Row functions allow specifying column names for dataframes

# In[51]:


data = sc.parallelize([Row(id=1,
                           name="Alice",
                           score=50)])
data


# In[53]:


data.count()


# In[52]:


data.collect()


# In[54]:


df = data.toDF()
df.show()


# #### Working with multiple rows

# In[66]:


data = sc.parallelize([Row(
                           id=1,
                           name="Alice",
                           score=50
                        ),
                        Row(
                            id=2,
                            name="Bob",
                            score=80
                        ),
                        Row(
                            id=3,
                            name="Charlee",
                            score=75
                        )])


# In[67]:


df = data.toDF()
df.show()


# #### Multiple columns with complex data types

# In[71]:


complex_data = sc.parallelize([Row(
                                col_float=1.44,
                                col_integer=10,
                                col_string="John")
                           ])


# In[72]:


complex_data_df = complex_data.toDF()
complex_data_df.show()


# In[73]:


complex_data = sc.parallelize([Row(
                                col_float=1.44, 
                                col_integer=10, 
                                col_string="John", 
                                col_boolean=True, 
                                col_list=[1, 2, 3])
                           ])


# In[74]:


complex_data_df = complex_data.toDF()
complex_data_df.show()


# In[79]:


complex_data = sc.parallelize([Row(
                                col_list = [1, 2, 3], 
                                col_dict = {"k1": 0, "k2": 1, "k3": 2}, 
                                col_row = Row(columnA = 10, columnB = 20, columnC = 30), 
                                col_time = datetime(2014, 8, 1, 14, 1, 5)
                            )])


# In[80]:


complex_data_df = complex_data.toDF()
complex_data_df.show()


# #### Multiple rows with complex data types

# In[89]:


complex_data = sc.parallelize([Row(
                                col_list = [1, 2, 3],
                                col_dict = {"k1": 0},
                                col_row = Row(a=10, b=20, c=30),
                                col_time = datetime(2014, 8, 1, 14, 1, 5)
                            ),              
                            Row(
                                col_list = [1, 2, 3, 4, 5], 
                                col_dict = {"k1": 0,"k2": 1 }, 
                                col_row = Row(a=40, b=50, c=60),
                                col_time = datetime(2014, 8, 2, 14, 1, 6)
                            ),
                            Row(
                                col_list = [1, 2, 3, 4, 5, 6, 7], 
                                col_dict = {"k1": 0, "k2": 1, "k3": 2 }, 
                                col_row = Row(a=70, b=80, c=90),
                                col_time = datetime(2014, 8, 3, 14, 1, 7)
                            )]) 


# In[90]:


complex_data_df = complex_data.toDF()
complex_data_df.show()


# #### Creating DataFrames using SQLContext
# 
# * SQLContext can create dataframes directly from raw data

# In[92]:


sqlContext = SQLContext(sc)


# In[93]:


sqlContext


# In[99]:


df = sqlContext.range(5)
df


# In[100]:


df.show()


# In[101]:


df.count()


# #### Rows specified in tuples

# In[104]:


data = [('Alice', 50),
        ('Bob', 80),
        ('Charlee', 75)]


# In[105]:


sqlContext.createDataFrame(data).show()


# In[106]:


sqlContext.createDataFrame(data, ['Name', 'Score']).show()


# In[200]:


complex_data = [
                 (1.0,
                  10,
                  "Alice", 
                  True, 
                  [1, 2, 3], 
                  {"k1": 0},
                  Row(a=1, b=2, c=3), 
                  datetime(2014, 8, 1, 14, 1, 5)),

                 (2.0,
                  20,
                  "Bob", 
                  True, 
                  [1, 2, 3, 4, 5], 
                  {"k1": 0,"k2": 1 }, 
                  Row(a=1, b=2, c=3), 
                  datetime(2014, 8, 1, 14, 1, 5)),

                  (3.0,
                   30,
                   "Charlee", 
                   False, 
                   [1, 2, 3, 4, 5, 6], 
                   {"k1": 0, "k2": 1, "k3": 2 }, 
                   Row(a=1, b=2, c=3), 
                   datetime(2014, 8, 1, 14, 1, 5))
                ] 


# In[201]:


sqlContext.createDataFrame(complex_data).show()


# In[202]:


complex_data_df = sqlContext.createDataFrame(complex_data, [
        'col_integer',
        'col_float',
        'col_string',
        'col_boolean',
        'col_list',
        'col_dictionary',
        'col_row',
        'col_date_time']
    )
complex_data_df.show()


# #### Creating dataframes using SQL Context and the Row function
# * Row functions can be used without specifying column names

# In[203]:


data = sc.parallelize([
    Row(1, "Alice", 50),
    Row(2, "Bob", 80),
    Row(3, "Charlee", 75)
])


# In[204]:


column_names = Row('id', 'name', 'score')  
students = data.map(lambda r: column_names(*r))


# In[205]:


students


# In[206]:


students.collect()


# In[207]:


students_df = sqlContext.createDataFrame(students)
students_df


# In[208]:


students_df.show()


# #### Extracting specific rows from dataframes

# In[209]:


complex_data_df.first()


# In[210]:


complex_data_df.take(2)


# #### Extracting specific cells from dataframes

# In[211]:


cell_string = complex_data_df.collect()[0][2]
cell_string


# In[212]:


cell_list = complex_data_df.collect()[0][4]
cell_list


# In[213]:


cell_list.append(100)
cell_list


# In[214]:


complex_data_df.show()


# #### Selecting specific columns

# In[215]:


complex_data_df.rdd    .map(lambda x: (x.col_string, x.col_dictionary))    .collect()


# In[216]:


complex_data_df.select(
    'col_string',
    'col_list',
    'col_date_time'
).show()


# #### Editing columns

# In[217]:


complex_data_df.rdd           .map(lambda x: (x.col_string + " Boo"))           .collect()


# #### Adding a column

# In[218]:


complex_data_df.select(
                   'col_integer',
                   'col_float'
            )\
           .withColumn(
                   "col_sum",
                    complex_data_df.col_integer + complex_data_df.col_float
           )\
           .show()


# In[220]:


complex_data_df.select('col_boolean')               .withColumn(
                   "col_opposite",
                   complex_data_df.col_boolean == False )\
               .show()


# #### Editing a column name

# In[225]:


complex_data_df.withColumnRenamed("col_dictionary","col_map").show()


# In[226]:


complex_data_df.select(complex_data_df.col_string.alias("Name")).show()


# #### Interoperablity between Pandas dataframe and Spark dataframe

# In[232]:


import pandas


# In[234]:


df_pandas = complex_data_df.toPandas()
df_pandas


# In[235]:


df_spark = sqlContext.createDataFrame(df_pandas).show()  
df_spark

