from pyspark.sql.types import *
from pyspark.sql import SparkSession


if __name__ == "__main__":

    # Set your local host to be the master node of your cluster
    # Set the appName for your Spark session
    # Join session for app if it exists, else create a new one
    sparkSession = SparkSession.builder.master("local")\
                              .appName("SparkStreamingAppendMode")\
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
    fileStreamDF = sparkSession.readStream\
                               .option("header", "true")\
                               .schema(schema)\
                               .csv("../datasets/droplocation")


    # Check whether input data is streaming or not
    print(" ")
    print("Is the stream ready?")
    print(fileStreamDF.isStreaming)


    # Print Schema
    print(" ")
    print("Schema of the input stream: ")
    print(fileStreamDF.printSchema)


    # Create a trimmed version of the input dataframe with specific columns
    # We cannot sort a DataFrame unless aggregate is used, so no sorting here
    trimmedDF = fileStreamDF.select(
                                      fileStreamDF.borough,
                                      fileStreamDF.year,
                                      fileStreamDF.month,
                                      fileStreamDF.value
                                      )\
                             .withColumnRenamed(
                                      "value",
                                      "convictions"
                                      )


    # We run in append mode, so only new rows are processed,
    # and existing rows in Result Table are not affected
    # The output is written to the console
    # We set truncate to false. If true, the output is truncated to 20 chars
    # Explicity state number of rows to display. Default is 20
    query = trimmedDF.writeStream\
                      .outputMode("append")\
                      .format("console")\
                      .option("truncate", "false")\
                      .option("numRows", 30)\
                      .start()\
                      .awaitTermination()








