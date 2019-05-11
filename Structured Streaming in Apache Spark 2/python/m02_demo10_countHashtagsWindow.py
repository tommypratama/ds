import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.sql.functions import udf
from pyspark.sql.functions import window
from pyspark.sql.types import *

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: spark-submit m02_demo10_countHashtagsWindow.py <hostname> <port>", 
                file=sys.stderr)
        exit(-1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    spark = SparkSession\
        .builder\
        .appName("HashtagCount")\
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    lines = spark\
        .readStream\
        .format("socket")\
        .option("host", host)\
        .option("port", port)\
        .option("includeTimestamp", "true")\
        .load()

    words = lines.select(
        explode(
            split(lines.value, " "))\
        .alias("word"), 
        lines.timestamp)

    def extract_tags(word):
        if word.lower().startswith("#"):
            return word
        else:
            return "nonTag"

    extract_tags_udf = udf(extract_tags, StringType())
    resultDF = words.withColumn("tags", extract_tags_udf(words.word))

    windowedHashtagCounts = resultDF.where(resultDF.tags != "nonTag")\
                                     .groupBy(
                                            window(resultDF.timestamp, 
                                                    "50 seconds", 
                                                    "30 seconds"),
                                            resultDF.tags)\
                                     .count()\
                                     .orderBy("count", ascending=False)

    query = windowedHashtagCounts.writeStream\
                                  .outputMode("complete")\
                                  .format("console")\
                                  .option("truncate", "false")\
                                  .start()\
                                  .awaitTermination()




