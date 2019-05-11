#### m01_demo01

# Start netcat utility
nc -l 9999
nc localhost 9999

# Run Python code to read stream from socket and write to console
spark-submit m01_demo01_netcat.py localhost 9999

# Input sentences
Baked chicken is healthier than fried chicken
I prefer fried fish over fried chicken


#### m02_demo01

# Start streaming app 
spark-submit m02_demo01-appendMode.py
spark-submit m02_demo02_completeMode.py
spark-submit am02_demo03_aggregateSum.py
spark-submit m02_demo04_sqlQuery.py
spark-submit m02_demo05_addTimestamp.py
spark-submit m02_demo06_groupbyTimestamp.py
spark-submit m02_demo07_window.py
spark-submit m03_demo02_countHashtags.py
spark-submit m03_demo03_countHashtagsWindow.py 
spark-submit m03_twitterStreaming.py
spark-submit m04_demo02_kafkaHashtagProducer.py
spark-submit m04_demo03_tweetSentiment.py
spark-submit m04_demo04_tweetSentimentCount.py
spark-submit m04_kafkaTweetProducer.py

# Drop one file at a time into the datasets/droplocation directory


#### m02_demo02

# Start streaming app 
spark-submit m02_demo02-completeMode.py 

# The datasets/droplocation directory should have 3-5 files already from previous lab
# Drop three files at once into the datasets/droplocation directory

#### m02_demo03



#### m03_demo01

# Start Twitter streaming app
python m03_twitterStreaming.py localhost 9999 fifa nba ipl
python m03_twitterStreaming.py localhost 9999 "world cup" nba ipl

# Start Spark Streaming app
spark-submit m03_demo01_countHashtags.py localhost 9999



#### m04_demo01

# Start Zookeeper
zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties

# Start Kafka server
kafka-server-start.sh $KAFKA_HOME/config/server.properties

# CEATE A KAFKA TOPIC called first_kafka_topic
kafka-topics.sh --create --zookeeper localhost:2181 \
--replication-factor 1 \
--partitions 1 \
--topic first_kafka_topic

# KAFKA TOPICS LIST
# Port 2181 for zookeeper is set in the zookeeper.properties file
kafka-topics.sh --list --zookeeper localhost:2181

# KAFKA PRODUCER
kafka-console-producer.sh --broker-list localhost:9092 \
--topic first_kafka_topic

# KAFKA CONSUMER
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
--topic first_kafka_topic \
--from-beginning

# KAFKA DELETE A TOPIC
kafka-topics.sh --delete --zookeeper localhost:2181 --topic first_kafka_topic



#### m04_demo02

# Create a Kafka topic on which to publish tweets
kafka-topics.sh --create --zookeeper localhost:2181 \
--replication-factor 1 \
--partitions 1 \
--topic tweets_topic

# Run Kafka Tweet Producer
python m04_kafkaTweetProducer.py localhost 9092 tweets_topic "world cup"


#### m04_demo03

# Run Kafka Consumer
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.2.0 \
m04_demo04_tweetSentimentCount.py localhost 9092 tweets_topic






