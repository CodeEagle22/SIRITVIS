# Copyright (c) [year] [your name]
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from apify_client import ApifyClient
import pickle
import pandas as pd
from datetime import datetime
import warnings
from instagram_private_api import Client
import numpy as np
# Suppress warnings
warnings.filterwarnings("ignore")

# Adjust log level to suppress log messages
import logging
logging.getLogger().setLevel(logging.ERROR)

class InstagramStreamer:
    def __init__(self, api_token, save_path, instagram_username, instagram_password, hashtags = ['instagram'], limit = 100):


        """
        Initialize the RedditStreamer object.

        Args:
            api_token (str): API token for the apify API.

            save_path (str): Path to save the scraped data.

            instagram_username (str): Your Instagram username.

            instagram_password (str): Your Instagram password.

            hashtags (list): Keywords to filter the posts (default: 'instagram').

            limit (int): Total number of posts to extract (default: 20).
        """
        assert isinstance(api_token, str), "api_token should be a string."
        assert isinstance(save_path, str), "save_path should be a string."
        assert isinstance(instagram_username, str), "instagram_username should be a string."
        assert isinstance(instagram_password, str), "instagram_password should be a string."
        assert isinstance(hashtags, list), "hashtags should be a list, default hashtags='instagram'."
        assert isinstance(limit, int), "limit should be a int, default limit=20."

        self.client = ApifyClient(api_token)
        self.api = Client(instagram_username, instagram_password)
        self.hashtags = hashtags
        self.limits = limit 
        self.save_path = save_path
        
    def run_scraper(self):
        run_input = {
            "hashtags": self.hashtags,
            "resultsLimit": self.limits,
        }
        run = self.client.actor("apify/instagram-hashtag-scraper").call(run_input=run_input)
        
        storage = []
        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            storage.append(item)
            
        
        df = pd.DataFrame(storage)

        

        latitude = []
        longitude = []
        for index, row in enumerate(df.iloc):
            
            try:
                if pd.isna(row['locationId']):
                    latitude.append(np.nan)
                    longitude.append(np.nan)
                else:
                    data = self.api.location_info(int(row['locationId']))
                    latitude.append(data['location']['lat'])
                    longitude.append(data['location']['lng'])
            except:
                latitude.append(np.nan)
                longitude.append(np.nan)

        df['center_coord_X'] = longitude
        df['center_coord_Y'] = latitude
        df.rename(columns={
                'caption': 'text',
                'timestamp': 'created_at',
            }, inplace=True)
        # Write data to a pickle file
        now = datetime.now()
        
        strzeit = now.strftime('%Y%m%d_%H%M%S')
        file_name = self.save_path + 'instagram_{}.pkl'.format(strzeit)
        with open(file_name, 'wb') as f:
            pickle.dump(df, f)
    



    def run(self):
        try:
            self.run_scraper()
        except KeyboardInterrupt:
            print("Streaming interrupted at:", datetime.now().strftime('%Y%m%d-%H%M%S'))
        except Exception as e:
            print("Error occurred: ", e)
        finally:
            return self.save_path


