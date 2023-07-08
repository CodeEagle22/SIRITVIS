
import pickle

from datetime import datetime


import pandas as pd

import re

import warnings

import praw
from langdetect import detect







# 1. Data Scraping: SocialMediaStreamer class

# Contents of the class "SocialMediaStreamer"
# - Streaming of social media data via the respective social media API.
# - Results saved in a specified format (e.g., JSON, PKL).

# Additional Information:
# To access social media data, one typically needs to have API access credentials provided by the social media platform.
# These credentials usually include a client ID, client secret, and user agent, which are required for authentication.

# Attention!
# It is advised to keep the StreamListener object as sparse as possible to reduce computing time and prevent
# potential errors. Depending on the API, there may be limitations on the number of requests per minute or other
# rate limits that need to be taken into account to avoid disruptions in the streaming process.

# Set up the StreamListener object. For more information,
# refer to the documentation of the specific social media platform's API.

# __Details about the class 'SocialMediaStreamer':__
# The data handling method is typically called "on_data". It is used to access each individual social media post
# or comment in its raw form. Operations can be performed directly on the incoming raw data. For example, filtering
# based on specific attributes, patterns, or keywords can be applied to process only relevant posts or comments.

# Another method of the class can handle rate limits to avoid exceeding the allowed number of requests within a given time frame.
# If the API object is set to wait_on_rate_limit = True, this method may not be necessary.

# The "on_error" method can handle any other errors that might occur during the streaming process.
# Error codes and timestamps can be logged, and the stream can be terminated if necessary.

# The streaming API of the respective social media platform is accessed by passing the incoming stream (posts or comments)
# through an instantiated stream object. A default API-provided class, such as StreamListener, may be inherited,
# and modifications can be made in the derived class (e.g., SocialMediaStreamer) to specify filtering conditions or
# other processing requirements.

# When a listener object from the SocialMediaStreamer class is instantiated, the object can open a file to store
# the streamed data. The file can be named based on the current timestamp to associate it with the start time of
# the streaming session.

# Note:
# Each social media platform may have its own specific guidelines, terms of service, and data usage policies that
# need to be followed when accessing and using their data through the API. It is important to review and comply with
# the respective platform's documentation and policies to ensure legal and ethical usage of the data.




class RedditStreamer():
    def __init__(self, client_id, client_secret, user_agent, save_path, keywords='all', subreddit_name='all'):
        
        """
        Initialize the RedditScraper object.

        Args:

            client_id (str): Client ID for the Reddit API.

            client_secret (str): Client Secret for the Reddit API.

            user_agent (str): User Agent for the Reddit API.

            save_path (str): Path to save the scraped data.

            keywords (str or list or 'all'): Keywords to filter the posts or 'all' to retrieve all posts (default: 'all').

            subreddit_name (str or 'all'): Name of the subreddit to scrape or 'all' to scrape all subreddits (default: 'all').

        """

        assert isinstance(client_id, str), "client_id should be a string."
        assert isinstance(client_secret, str), "client_secret should be a string."
        assert isinstance(user_agent, str), "user_agent should be a string."
        assert isinstance(save_path, str), "save_path should be a string."
        assert isinstance(keywords, (str, list)) or keywords == 'all', "keywords should be a string, a list, or 'all'."
        assert isinstance(subreddit_name, (str, list)) or subreddit_name == 'all', "subreddit_name should be a string, a list, or 'all'."

        
        self.subreddit_name = subreddit_name
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.keywords = keywords
        self.save_path = save_path
        self.columns = ['title', 'score', 'author', 'id_str', 'subreddit', 'url', 'num_comments', 'text', 'created_at']
        self.reddit = None
        self.subreddit = None
        self.run()



    def connect_to_reddit(self):
        warnings.filterwarnings('ignore')
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )

        self.subreddit = self.reddit.subreddit(self.subreddit_name)


    def clean_text(self,text):
        # Remove hyperlinks starting with http or https
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'https\S+', '', text)

        # Remove newline characters
        text = text.replace('\n', ' ')

        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)

        return text

    def scrape_data(self):
        data = []
        now = datetime.now()
        print('Start streaming: ' + now.strftime('%Y%m%d-%H%M%S'))
        strzeit = now.strftime('%Y%m%d-%H%M%S')

        for keyword in self.keywords:
            warnings.filterwarnings('ignore')
            # Use Reddit API to search for posts containing the keyword in the subreddit
            posts = self.subreddit.search(keyword, limit=None)

            # Extract the desired data from the posts
            for post in posts:
                if post.selftext != "":
                    data.append([
                        self.clean_text(str(post.title)),
                        post.score,
                        str(post.author.name),
                        str(post.id),
                        str(post.subreddit),
                        str(post.url),
                        str(post.num_comments),
                        self.clean_text(str(post.selftext)),
                        str(post.created)
                    ])

        df = pd.DataFrame(data, columns=self.columns)

        # Chunk dataframe to reduce memory usage
        chunk_size = 90000000
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        # Write chunks to separate pickle files
        for i, chunk in enumerate(chunks):
            file_name = self.save_path+'reddit_{}_{}.pkl'.format(strzeit, i)
            with open(file_name, 'wb') as f:
                pickle.dump(chunk, f)


    def run(self):
        self.connect_to_reddit()
        while True:
            try:
                self.scrape_data()
            except KeyboardInterrupt:
                print("Stopped at: " + datetime.now().strftime('%Y%m%d-%H%M%S'))
                break
            except Exception as e:
                print("Error occurred: ", e)
                continue
