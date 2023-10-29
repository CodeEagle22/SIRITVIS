# Copyright (c) [year] [your name]
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pickle
from datetime import datetime
import pandas as pd
import re
import os
import warnings
from textblob import TextBlob
import praw

import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

# Adjust log level to suppress log messages
import logging
logging.getLogger().setLevel(logging.ERROR)

class RedditStreamer():
    def __init__(self, client_id, client_secret, user_agent, save_path, keywords='all', subreddit_name='all'):
        """
        Initialize the RedditStreamer object.

        Args:
            client_id (str): Client ID for the Reddit API.
            client_secret (str): Client Secret for the Reddit API.
            user_agent (str): User Agent for the Reddit API.
            save_path (str): Path to save the scraped data.
            keywords (str or list or 'all'): Keywords to filter the posts or 'all' to retrieve all posts (default: 'all').
            subreddit_name (str or 'all'): Name of the subreddit to scrape or 'all' to scrape all subreddits (default: 'all').
            min_file_size_mb (int): Minimum file size in MB to store in one file (default: 1).
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
        self.min_file_size = 1 * 1024 * 1024  # Convert to bytes
        self.data_buffer = []
        self.current_file_size = 0
        self.run()

    def connect_to_reddit(self):
        warnings.filterwarnings('ignore')
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )
        self.subreddit = self.reddit.subreddit(self.subreddit_name)

    def clean_text(self, text):
        # Remove hyperlinks starting with http or https
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'https\S+', '', text)

        # Remove newline characters
        text = text.replace('\n', ' ')

        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)

        return text

    def scrape_data(self):
        now = datetime.now()
        print('Start streaming: ' + now.strftime('%Y%m%d-%H%M%S'))
        strzeit = now.strftime('%Y%m%d-%H%M%S')

        for keyword in self.keywords:
            warnings.filterwarnings('ignore')
            # Use Reddit API to search for posts containing the keyword in the subreddit
            posts = self.subreddit.search(keyword, limit=None)

            # Extract the desired data from the posts
            for post in posts:
                if post.selftext != '' and TextBlob(post.selftext).sentiment.polarity < 0:
                    data = [
                        self.clean_text(str(post.title)),
                        post.score,
                        str(post.author.name),
                        str(post.id),
                        str(post.subreddit),
                        str(post.url),
                        str(post.num_comments),
                        self.clean_text(str(post.selftext)),
                        str(post.created)
                    ]
                    self.data_buffer.append(data)
                    self.current_file_size += len(str(data).encode('utf-8'))

                    if self.current_file_size >= self.min_file_size:
                        self.write_to_file(strzeit)
                        self.current_file_size = 0
                        self.data_buffer = []

        # Write any remaining data to a file if not already written
        if self.data_buffer:
            self.write_to_file(strzeit)

    def write_to_file(self, timestamp):
        df = pd.DataFrame(self.data_buffer, columns=self.columns)
        # Write data to a pickle file
        file_name = self.save_path + 'reddit_{}.pkl'.format(timestamp)
        with open(file_name, 'wb') as f:
            pickle.dump(df, f)

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
        return self.save_path
