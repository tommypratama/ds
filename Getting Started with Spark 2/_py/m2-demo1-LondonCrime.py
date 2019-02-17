
# coding: utf-8

# ### Analyzing London crime statistics

# #### Creating a Spark session
# 
# * Encapsulates SparkContext and the SQLContext within it

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName("Analyzing London crime data")     .getOrCreate()


# #### Reading external data as a dataframe

# In[2]:


data = spark.read            .format("csv")            .option("header", "true")            .load("../datasets/london_crime_by_lsoa.csv")


# In[3]:


data.printSchema()


# In[4]:


data.count()


# In[5]:


data.limit(5).show()


# #### Cleaning data
# * Drop rows which do not have valid values
# * Drop columns which we do not use in our analysis

# In[6]:


data.dropna()


# In[7]:


data = data.drop("lsoa_code")

data.show(5)


# #### Boroughs included in the report

# In[8]:


total_boroughs = data.select('borough')                     .distinct()        
total_boroughs.show()


# In[9]:


total_boroughs.count()


# In[10]:


hackney_data = data.filter(data['borough'] == "Hackney")

hackney_data.show(5)


# In[11]:


data_2015_2016 = data.filter(data['year'].isin(["2015", "2016"]))

data_2015_2016.sample(fraction=0.1).show()


# In[12]:


data_2014_onwards = data.filter(data['year'] >= 2014 )

data_2014_onwards.sample(fraction=0.1).show()


# #### Total crime per borough

# In[13]:


borough_crime_count = data.groupBy('borough')                          .count()
    
borough_crime_count.show(5)


# #### Total convictions per borough

# In[14]:


borough_conviction_sum = data.groupBy('borough')                             .agg({"value":"sum"})

borough_conviction_sum.show(5)


# In[15]:


borough_conviction_sum = data.groupBy('borough')                             .agg({"value":"sum"})                             .withColumnRenamed("sum(value)","convictions")

borough_conviction_sum.show(5)


# #### Per-borough convictions expressed in percentage

# Total convictions

# In[16]:


total_borough_convictions = borough_conviction_sum.agg({"convictions":"sum"})

total_borough_convictions.show()


# Extracting total convictions into a variable

# In[17]:


total_convictions = total_borough_convictions.collect()[0][0]


# A new column which contains the % convictions for each borough

# In[18]:


import pyspark.sql.functions as func


# In[19]:


borough_percentage_contribution = borough_conviction_sum.withColumn(
    "% contribution",
    func.round(borough_conviction_sum.convictions / total_convictions * 100, 2))

borough_percentage_contribution.printSchema()


# In[20]:


borough_percentage_contribution.orderBy(borough_percentage_contribution[2].desc())                               .show(10)


# #### Convictions across months in a particular year

# In[21]:


conviction_monthly = data.filter(data['year'] == 2014)                         .groupBy('month')                         .agg({"value":"sum"})                         .withColumnRenamed("sum(value)","convictions")


# In[22]:


total_conviction_monthly = conviction_monthly.agg({"convictions":"sum"})                                             .collect()[0][0]

total_conviction_monthly = conviction_monthly    .withColumn("percent",
                func.round(conviction_monthly.convictions/total_conviction_monthly * 100, 2))
total_conviction_monthly.columns


# In[23]:


total_conviction_monthly.orderBy(total_conviction_monthly.percent.desc()).show()


# #### Most prevalant crimes

# In[24]:


crimes_category = data.groupBy('major_category')                      .agg({"value":"sum"})                      .withColumnRenamed("sum(value)","convictions")


# In[25]:


crimes_category.orderBy(crimes_category.convictions.desc()).show()


# In[26]:


year_df = data.select('year')


# In[27]:


year_df.agg({'year':'min'}).show()


# In[28]:


year_df.agg({'year':'max'}).show()


# In[29]:


year_df.describe().show()


# In[34]:


data.crosstab('borough', 'major_category')    .select('borough_major_category', 'Burglary', 'Drugs', 'Fraud or Forgery', 'Robbery')    .show()


# #### Visualizing the data

# In[81]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# #### Distribution of crime across boroughs in a particular year

# In[92]:


def describe_year(year):
    yearly_details = data.filter(data.year == year)                         .groupBy('borough')                         .agg({'value':'sum'})                         .withColumnRenamed("sum(value)","convictions")
    
    borough_list = [x[0] for x in yearly_details.toLocalIterator()]
    convictions_list = [x[1] for x in yearly_details.toLocalIterator()]
  
    plt.figure(figsize=(33, 10)) 
    plt.bar(borough_list, convictions_list)
    
    plt.title('Crime for the year: ' + year, fontsize=30)
    plt.xlabel('Boroughs',fontsize=30)
    plt.ylabel('Convictions', fontsize=30)

    plt.xticks(rotation=90, fontsize=30)
    plt.yticks(fontsize=30)
    plt.autoscale()
    plt.show()


# In[93]:


describe_year('2014')

