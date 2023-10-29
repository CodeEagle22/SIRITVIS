# Copyright (c) [year] [your name]
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import collections
from datetime import datetime
import json
import math
import os
import time
import tweepy as tw
from tweepy import OAuthHandler
from tweepy import API
from tweepy.streaming import StreamListener
from tweepy import Stream
from urllib3.exceptions import ProtocolError, ReadTimeoutError
from textblob import TextBlob
import warnings
import math

# Suppress warnings
warnings.filterwarnings("ignore")

# Adjust log level to suppress log messages
import logging
logging.getLogger().setLevel(logging.ERROR)

class TwitterStreamer(StreamListener):
    def __init__(self, consumer_key, consumer_secret,access_token,access_secret, languages, locations, save_path=os.getcwd(), extended=True, keywords=[], hashtag=True):
        
        """
        Initialize the TweetMapper object.

        Args:
            consumer_key (str): consumer_key for the Twitter API.
            consumer_secret (str): consumer_secret for the Twitter API.
            access_token (str): access_token for the Twitter API.
            access_secret (str): access_secret for the Twitter API.
            languages (list): List of languages to filter tweets. e.g ['en']
            locations (list): Box coordinates of locations to filter tweets. e.g [51.416016,5.528511,90.966797,34.669359]
            save_path (str): Path to where the tweets will be saved (default: current working directory).
            extended (bool): Flag to determine whether to retrieve extended tweets (default: True).
            keywords (list): List of keywords to track (default: empty list).
            hashtag (bool): Flag to determine whether to save tweets with hashtags (default: True).
        """

        assert isinstance(consumer_key, str), "consumer_key must be a string"
        assert isinstance(consumer_secret, str), "consumer_secret must be a string"
        assert isinstance(access_token, str), "access_token must be a string"
        assert isinstance(access_secret, str), "access_secret must be a string"
        assert isinstance(languages, list), "languages must be a list"
        assert isinstance(locations, list), "locations must be a list"
        assert isinstance(save_path, str), "save_path must be a string"
        assert isinstance(extended, bool), "extended must be a boolean"
        assert isinstance(keywords, list), "keywords must be a list"
        assert isinstance(hashtag, bool), "hashtag must be a boolean"

        super(StreamListener, self).__init__()
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_secret = access_secret
        self.api = self.access_auth()  # Access Twitter API using authentication credentials
        self.extended = extended  # Flag to determine whether to retrieve extended tweets
        self.hashtag = hashtag  # Flag to determine whether to save tweets with hashtags
        self.languages = languages  # List of languages to filter tweets
        self.locations = locations  # List of locations to filter tweets
        self.keywords = keywords  # List of keywords to track
        self.save_path = save_path
        now = datetime.now()
        self.save_file = open(os.path.join(self.save_path, 'tweets ' + now.strftime('%Y%m%d-%H%M%S') + '.json'), 'w')
        self.streaming()  # Start streaming tweets
        

    def access_auth(self):
        

        
        # Set up OAuthHandler with authentication credentials
        auth = OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_secret)

        # Create API object using authentication
        api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        return api
        

    def stream_process(self):
        self.stream = tw.Stream(auth=self.api.auth, listener=self, tweet_mode='extended')
        now = datetime.now()
        print('Start streaming: ' + now.strftime('%Y%m%d-%H%M%S'))

        # Start streaming and filter tweets based on keywords, languages, and locations
        self.stream.filter(track=self.keywords, languages=self.languages, locations=self.locations)

    def streaming(self):
        while True:
            try:
                self.stream_process()
                return self.save_path
            except KeyboardInterrupt:
                now = datetime.now()
                print("Stopped at: " + now.strftime('%Y%m%d-%H%M%S'))
                self.stream.disconnect()
                return False
            except ProtocolError:
                now = datetime.now()
                print('Incomplete read error - too many tweets were posted live at the same time at your location!' +
                      now.strftime('%Y%m%d-%H%M%S'))
                self.stream.disconnect()
                time.sleep(60)
                continue
            except ReadTimeoutError:
                self.stream.disconnect()
                now = datetime.now()
                print(now.strftime('%Y%m%d-%H%M%S') + ': ReadTimeoutError exception! Check your internet connection!')
                return False
        

    def on_data(self, tweet):
        # Save tweets based on specified conditions
        if TextBlob(tweet).sentiment.polarity<0:
          
          if 'extended_tweet' in tweet and self.extended and '#' in tweet and self.hashtag:
              self.save_file.write(str(tweet))
          elif 'extended_tweet' in tweet and self.extended and not self.hashtag:
              self.save_file.write(str(tweet))
          elif not self.extended and '#' in tweet and self.hashtag:
              self.save_file.write(str(tweet))
          elif not self.extended and not self.hashtag:
              self.save_file.write(str(tweet))

    def on_limit(self, status_code):
        if status_code == 420:
            now = datetime.now()
            print('API Rate limit reached: ' + now.strftime('%Y%m%d-%H%M%S'))
            return False

    def on_error(self, status_code):
        now = datetime.now()
        print(now.strftime('%Y%m%d-%H%M%S') + ' Error: ' + str(status_code) + '\nCheck out for more info: https://developer.twitter.com/en/support/twitter-api/error-troubleshooting')
        return False

    def on_timeout(self):
        print('Timeout: Wait 120 sec.')
        time.sleep(120)
        return

    
