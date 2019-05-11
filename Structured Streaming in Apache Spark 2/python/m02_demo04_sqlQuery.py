from pyspark.sql.types import *
from pyspark.sql import SparkSession


if __name__ == "__main__":

    # Set your local host to be the master node of your cluster
    # Set the appName for your Spark session
    # Join session for app if it exists, else create a new one
    sparkSession = SparkSession.builder.master("local")\
                              .appName("SparkStreamingSQLQuery")\
                              .getOrCreate()


    # ERROR log level will generate fewer lines of output compared to INFO and DEBUG                          
    sparkSession.sparkContext.setLogLevel("ERROR")


    # InferSchema not yet available in spark structured streaming 
    # (it is available in static dataframes)
    # We explicity state the schema of the input data
    schema = StructType([StructField("lsoa_code", StringType(), True),\
                         StructField("borough", StringType(), True),\
                         StructField("major_category", StringType(), True),\
                         StructField("minor_category", StringType(), True),\
                         StructField("value", StringType(), True),\
                         StructField("year", StringType(), True),\
                         StructField("month", StringType(), True)])


    # Read stream into a dataframe
    # Since the csv data includes a header row, we specify that here
    # We state the schema to use and the location of the csv files
    # maxFilesPerTrigger sets the number of new files to be considered in each trigger
    fileStreamDF = sparkSession.readStream\
                               .option("header", "true")\
                               .option("maxFilesPerTrigger", 2)\
                               .schema(schema)\
                               .csv("../datasets/droplocation")

    
    # Registering Table
    # Create a view which can later be queried like a table
    fileStreamDF.createOrReplaceTempView("LondonCrimeData")


    # Using SQL query
    # We use the LondonCrimeData view like a table
    # We only select the crime category and numConvictions columns
    # We narrow down the data to only include crimes committed in 2016
    categoryDF = sparkSession.sql("SELECT major_category, value \
                                    FROM LondonCrimeData \
                                    WHERE year = '2016'")



    # Use groupBy and agg functions to get total convictions per major_category
    # The new column created will be called sum(value) - rename to something meaningful
    # Order by number of convictions in descending order
    convictionsPerCategory = categoryDF.groupBy("major_category")\
                                      .agg({"value": "sum"})\
                                      .withColumnRenamed("sum(value)", "convictions")\
                                      .orderBy("convictions", ascending=False)


    # Write out our dataframe to the console
    query = convictionsPerCategory.writeStream\
                      .outputMode("complete")\
                      .format("console")\
                      .option("truncate", "false")\
                      .option("numRows", 30)\
                      .start()\
                      .awaitTermination()




