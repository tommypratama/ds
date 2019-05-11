import sys
import socket
import json
import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

class TweetsListener(StreamListener):

    def __init__(self, socket):

        print ("Tweets listener initialized")
        self.client_socket = socket

    def on_data(self, data):

        try:
            jsonMessage = json.loads(data)
            message = jsonMessage["text"].encode("utf-8")
            print (message)
            self.client_socket.send(message)

        except BaseException as e:
            print("Error on_data: %s" % str(e))

        return True

    def on_error(self, status):

        print (status)
        return True

def connect_to_twitter(connection, tracks):

    api_key = "N3BhfveA3Q4mXCtyRRphCrC7v"
    api_secret = "6UFnndGzuMwKKIldE2kw9syulo3Hb4gVwQ14umDP43jTdv9x1s"

    access_token = "368413415-5FaufzgFTozESYU00xp6nxrxoiUUp1IIhThVHHOp"
    access_token_secret = "c1JV2hFifKz2WFiLxLnpfrBPqmgeBLH5kBk5mqeq8PF9B"

    auth = OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_token_secret)

    twitter_stream = Stream(auth, TweetsListener(connection))
    twitter_stream.filter(track=tracks, languages=["en"])

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage: python m02_demo08_twitterStreaming.py <hostname> <port> <tracks>", 
                file=sys.stderr)
        exit(-1)

    host = sys.argv[1]
    port = int(sys.argv[2])
    tracks = sys.argv[3:]

    s = socket.socket()
    s.bind((host, port))

    print("Listening on port: %s" % str(port))

    s.listen(5)

    connection, client_address = s.accept()

    print( "Received request from: " + str(client_address))
    print("Initializing listener for these tracks: ", tracks)

    connect_to_twitter(connection, tracks)




















































































































































