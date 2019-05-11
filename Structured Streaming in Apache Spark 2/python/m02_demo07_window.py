from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
import time
import datetime
from pyspark.sql.functions import window


if __name__ == "__main__":

    
    # Set your local host to be the master node of your cluster
    # Set the appName for your Spark session
    # Join session for app if it exists, else create a new one
    sparkSession = SparkSession.builder.master("local")\
                              .appName("SparkStreamingGroupbyTimestampWindow")\
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

    # The User Defined Function (UDF)
    # Create a timestamp from the current time and return it
    def add_timestamp():
         ts = time.time()
         timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
         return timestamp


    # Register the UDF
    # Set the return type to be a String
    # A name is assigned to the registered function 
    add_timestamp_udf = udf(add_timestamp, StringType())


    # Create a new column called "timestamp" in fileStreamDF
    # Apply the UDF to every row in fileStreamDF - assign its return value to timestamp column
    fileStreamWithTS = fileStreamDF.withColumn("timestamp", add_timestamp_udf())


    # window(timeColumn, windowDuration, slideDuration=None, startTime=None)
    # timeColumn gives the time field to use when creating a window
    # windowDuration gives the length of the window
    # slideDuration is the gap between each window (Windows can overlap)
    # slideDuration must be <= windowDuration
    # The #convictions for a particular window will likely increase with each batch of files processed - 
    # this is because more timestamps within that window will be encountered in the new batch
    windowedCounts = fileStreamWithTS.groupBy(
                                        window(fileStreamWithTS.timestamp, 
                                                "30 seconds", 
                                                "18 seconds"))\
                                      .agg({"value": "sum"})\
                                      .withColumnRenamed("sum(value)", "convictions")\
                                      .orderBy('convictions', ascending=False)


    # Write output to the console
    query = windowedCounts.writeStream\
                          .outputMode("complete")\
                          .format("console")\
                          .option("truncate","false")\
                          .start()\
                          .awaitTermination()