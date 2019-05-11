import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split


# Check that correct number of args have been passed as input
if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: spark-submit m01_demo01_netcat.py <hostname> <port>", file=sys.stderr)
        exit(-1)


    # Extract host and port from args    
    host = sys.argv[1]
    port = int(sys.argv[2])

    
    # Set the app name when creating a Spark session
    # If a Spark session is already created for the app, use that. 
    # Else create a new session for that app
    spark = SparkSession\
        .builder\
        .appName("NetcatWordCount")\
        .getOrCreate()


    # Set log level. Use ERRROR to reduce the amount of output seen
    spark.sparkContext.setLogLevel("ERROR")


    # Create DataFrame representing the stream of input lines from connection to host:port
    # We're reading from the socket on the port where netcat is listening
    lines = spark\
        .readStream\
        .format('socket')\
        .option('host', host)\
        .option('port', port)\
        .load()


    # Split the lines into words
    # Explode turns each item in an array into a separate row
    # Alias sets the name of the column for the words
    # The result - each word of input is a row in a table with one column named "word"
    words = lines.select(
        explode(
            split(lines.value, ' ')
        ).alias('word')
    )


    # Generate running word count
    wordCounts = words.groupBy('word')\
                      .count()


    # Start running the query that prints the running counts to the console
    # Running in "complete" mode ensures that any operation uses ALL data 
    # - from previous and current batch 
    # The call to format sets where the stream is written to
    query = wordCounts.writeStream\
                      .outputMode('complete')\
                      .format('console')\
                      .start()

    query.awaitTermination()






