# Copyright (c) [year] [your name]
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pickle
import numpy as np
import collections
from datetime import datetime
from functools import partial
import glob
import itertools as it
import json
import math
import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.cli import download
from spacy.lang.en.stop_words import STOP_WORDS
import warnings
from wordcloud import WordCloud
import praw
from geopy.geocoders import Nominatim
import folium
from pandas import json_normalize
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

# Adjust log level to suppress log messages
import logging
logging.getLogger().setLevel(logging.ERROR)

class Cleaner(object):

    # arguments:
    # load_path (str): path containing the raw twitter json files
    # data_save_name (str): name of the data saved to drive after processing without file-suffix.
    # Default is 'my_cleaned_and_tokenized_data'
    # languages (list): List of string codes for certain languages to filter for. Default is None.
    # metadata (bool): Keep all covariates or only the ones necessary for the package. Default is 'False'
    # min_post_len (int): Refilter for an minimal token amount after the cleaning process. Default is None.
    # spacy_model (str): Choose the desired spacy model for text tokenization. Non-default model installation tutorial
    # and an overview about the supported languages can be found at https://spacy.io/usage/models.
    # Default is the small "English" model called 'en_core_web_sm'.
    def __init__(self, data_source, languages=None, metadata=False,
                 min_post_len=None, spacy_model='en_core_web_sm',data_source_type=None,file_format='csv',text_case=True):
        


        """
        Initialize the TopicModeling object.

        Args:
            data_source (str): Path to the streameed data to load or external csv data file path.

            languages (str or None): Languages to consider or None to consider all (default: None).

            metadata (bool): Whether to include metadata in the analysis (default: False).

            min_post_len (int or None): Minimum length of a post to consider or None to consider all (default: None).

            spacy_model (str): Name of the Spacy language model to load (default: 'en_core_web_sm').

            data_source_type (str): Data to clean as a 'twitter' or None if loading from file (default: None).

            file_format (str): Format of the data file to load (default: 'csv').

            external_data (str): CSV file path of external data source with 'text' column (default: None).

            text_case (bool): Whether to have text_tokens in the lowercase (default: True).

        """

        assert isinstance(data_source, str), "load_path should be a string."
        assert languages is None or isinstance(languages, list), "languages should be a list or None. ['en']"
        assert isinstance(metadata, bool), "metadata should be a boolean."
        assert min_post_len is None or (isinstance(min_post_len, int) and min_post_len > 0), "min_post_len should be a positive integer or None."
        assert isinstance(spacy_model, str), "spacy_model should be a string."
        assert data_source_type is None or isinstance(data_source_type, str), "data should be a 'twitter' or None."
        assert isinstance(file_format, str), "file_format should be a 'csv' or 'pkl'."
        assert isinstance(text_case, bool), "text_case should be a boolean."
        
        
        
        
        self.languages = languages
        self.load_path = data_source
        self.metadata = metadata
        self.min_post_len = min_post_len
        self.data = data_source_type
        self.file_format = file_format
        self.text_case = text_case
        try:
            self.spacy_model = spacy.load(spacy_model)
        except OSError:
            print('Downloading spacy language model. This will only happen once.')
            download(spacy_model)
            self.spacy_model = spacy.load(spacy_model)
        #self.spacy_model = spacy.load(spacy_model)  # loading the statistical spacy-model
        self.raw_data = self.loading()
        self.clean_data = self.cleaning()
        

    def loading(self):
        # All JSON files are read-in and merged together. Is was necessary to ensure that only
        # complete JSON strings were read. While streaming, it can sometimes happen that the stream stops during
        # the saving process of a tweet or that an error occurs. In that case, an incomplete JSON string would be saved,
        # which would lead to an error message. The script catches this error when reading-in the JSON files by
        # checking the code for each tweet, provided in JSON-string format, on whether the tweets string is complete or not.
        # If it is not, the incomplete string is ignored and the next one is read-in.
        # source: https://stackoverflow.com/questions/20400818/python-trying-to-deserialize-multiple-json-objects-in-a-file-with-each-object-s
        def process_data(data):
            if data == "twitter":
                return process_tweets()
            else:
                return process_pickles()

        def process_tweets():
            try:
            
                load_path = self.load_path  # Replace with the actual path to your tweet files
                if load_path.endswith('.csv'):
                    df_data = pd.read_csv(load_path)
                    df_data.reset_index(inplace=True)
                    df_data.rename(columns={'index': 'id_str'}, inplace=True)
                    self.data = None
                    return df_data
                else:
                    json_data = []
                    for filename in glob.glob(os.path.join(load_path, '*.json')):
                        try:
                            with open(filename, 'r') as f:
                                for line in f:
                                    while True:
                                        try:
                                            # Check if a JSON object is complete
                                            jfile = json.loads(line)
                                            break
                                        except ValueError:
                                            # Not yet a complete JSON value
                                            line += next(f)
                                    # Append the complete strings
                                    json_data.append(json.loads(line))
                        except Exception as e:
                            print(f"Error occurred while reading the file: {filename} - {str(e)}")
                            continue

                    if len(json_data) > 1500000:
                        print('Please read-in a maximum of 150,000 tweets per object!')
                        return False

                    df_data = pd.read_json(json.dumps(json_data))
                    

                    try:
                            df_data['place'] = df_data['place'].dropna().sample(n=len(json_data), replace=True).reset_index()['place']
                            return df_data
                    except:
                            self.data = None
                            return df_data
              
            except Exception as e:
                print("An error occurred:", e)
           

        def process_pickles():
            try: 
                load_path = self.load_path  # Replace with the actual path to your pickle files
                if load_path.endswith('.csv'):
                    df_data = pd.read_csv(load_path)
                    df_data.reset_index(inplace=True)
                    df_data.rename(columns={'index': 'id_str'}, inplace=True)
                    return df_data
                else:
                    pickle_data = []
                    for filename in glob.glob(os.path.join(load_path, '*.pkl')):
                        try:
                            with open(filename, 'rb') as f:
                                while True:
                                    try:
                                        obj = pickle.load(f)
                                        pickle_data.append(obj)
                                    except EOFError:
                                        break
                        except FileNotFoundError:
                            print("File not found: " + filename)
                            continue
                        except Exception as e:
                            print("Error occurred while reading the file: " + filename + " - " + str(e))
                            continue

                    df_data = pd.concat(pickle_data)
                    
                    
                    
                    return df_data
            except Exception as e:
                print("An error occurred:", e)
            
        return process_data(self.data)

    def is_text_in_language(text, language):
        try:
            detected_language = detect(text)
            return detected_language == language
        except:
            return False

    def cleaning(self):
        
        if self.raw_data is None or len(self.raw_data)==0:  # Check if the DataFrame is empty
                print("The DataFrame is empty. Check available files on given path.")

        else:
            if self.data == 'twitter':
                self.raw_data = self.raw_data.drop_duplicates('id')  # remove duplicates
                self.raw_data = self.raw_data[self.raw_data['is_quote_status'] == False]  # remove quoted statuses
                self.raw_data = self.raw_data[self.raw_data['retweeted'] == False]  # remove retweets
                if self.languages is not None:
                    for i in self.languages:
                        self.raw_data = self.raw_data[self.raw_data['lang'] == i]  # check for language

                
                # getting the indices, to check sub-json 'raw_data['place']':
                self.raw_data['place'] = self.raw_data['place'].fillna('')  # handling the "None"-values
                self.raw_data = self.raw_data[self.raw_data['place'] != '']  # take only tweets with bounding_box-geodata
                place_df = json_normalize(self.raw_data['place'])  # read the geo-location sub-json in as data frame

                poly_indices = place_df.index[place_df['bounding_box.type'] == 'Polygon'].to_numpy()  # check if location is
                # available and turn indices object to numpy array.


                # get a sub-df with the conditions above met:
                self.raw_data = self.raw_data.iloc[poly_indices, :]
                place_df = place_df.iloc[poly_indices, :]

                # if tweet is longer than 140 chars: The extended tweet-text is submitted to the 'text' column:
                indices_extw = np.array(self.raw_data[self.raw_data['extended_tweet']
                                        .notna()].index.tolist())  # get indices of extended tweets.
                ex_tweet_df = self.raw_data['extended_tweet']  # get the extended tweets sub-json
                ex_tweet_df = json_normalize(ex_tweet_df[indices_extw])  # normalize it
                ex_tweet_df = ex_tweet_df['full_text']  # save the full text as list
                ex_tweet_df = pd.Series(ex_tweet_df)
                ex_tweet_df = pd.Series(ex_tweet_df.values,
                                        index=indices_extw)  # change the list to a Series and attach the right indices.
                self.raw_data.loc[indices_extw, 'text'] = ex_tweet_df[indices_extw]  # overwrite the data in 'text',
                # in cases where the tweet is 'extended'.
            else:
                try:
                    self.raw_data = self.raw_data.drop_duplicates('id_str')  # remove duplicates
                except:
                    self.raw_data.reset_index(inplace=True)
                    self.raw_data.rename(columns={'index': 'id_str'}, inplace=True)
                    self.raw_data = self.raw_data.drop_duplicates('id_str')  # remove duplicates
                
                if self.languages is not None:
                    self.raw_data = self.raw_data[self.raw_data['text'].apply(lambda x: is_text_in_language(x, self.languages))] # check for language

                # split the string at the occurrence of the embedded hyperlink and take the first part over all
                # entries (remove hyperlinks):
                
                self.raw_data['text'] = self.raw_data['text'].astype(str).apply(lambda x: re.split('https://t.co', x)[0])


            # remove and append Emojis:
            if self.metadata:
                emojis = re.compile(
                    u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')  # emoji unicode
                indices = np.array(self.raw_data['text'].index.tolist())  # save the index numbers of the entries
                l1 = []
                for i in self.raw_data['text']:
                    l1.append(emojis.findall(i))
                l1 = pd.Series(l1)
                l1 = pd.Series(l1.values, index=indices)  # put the gathered values together with the old indices
                l1 = l1.rename('emojis')
                self.raw_data = pd.concat([self.raw_data, l1], axis=1)  # concat the series with the emojis to the dataframe

            self.raw_data['text'] = self.raw_data['text'].apply(
                lambda x: x.encode('ascii', 'ignore').decode('ascii'))  # remove emojis from textfield
            # remove mentions (usernames):
            self.raw_data['text'] = self.raw_data['text'].apply(
                lambda x: ' '.join(i for i in x.split() if not i.startswith('@')))

            # collect hashtags from the text:
            lists = self.raw_data['text'].str.split()  # split every word in the text at whitespace
            indices = np.array(lists.index.tolist())  # save the index numbers of the entries
            # make a new list and collect all hashtag-words:
            l1 = []
            for i in lists:
                l2 = []
                for j in i:
                    if j.startswith('#'):
                        a = re.split('[^#a-zA-Z0-9-]', j)  # remove all non-alphanumeric characters at end of hashtag
                        l2.append(a[0])
                l1.append(l2)

            l1 = pd.Series(l1)
            l1 = pd.Series(l1.values, index=indices)  # put the gathered values together with the old indices
            l1 = l1.rename('hashtags')
            self.raw_data = pd.concat([self.raw_data, l1], axis=1)  # concat the series to the dataframe
            self.raw_data['text'] = self.raw_data['text'].str.replace('#', '')
            
            if self.data == 'twitter':
                # append the location data:
                place_df = json_normalize(self.raw_data['place'])  # update 'place_df' to remaining numbers of tweets
                indices = np.array(self.raw_data.index.tolist())  # update indices
                st = place_df['bounding_box.coordinates'].apply(lambda x: str(x))  # convert all entries to strings
                st = pd.Series(st)  # list to series
                st = pd.Series(st.values, index=indices)  # insert updated indices
                st = st.str.replace('[', '')  # remove all unnecessary symbols
                st = st.str.replace(']', '')
                st = st.apply(lambda x: re.split(',', x))  # split the string to isolate each number
                st = pd.DataFrame(st)
                st = st.rename(columns={0: "bounding_box.coordinates_str"})  # rename the column

                # Calculate the center of the bounding box:
                # LONG FIRST; LAT LATER: center of rectangle for first entry: y1=st[0][1], y2=st[0][3], x1=st[0][0], x2=st[0][4]
                # xy-center: (x1+x2)/2, (y1+y2)/2
                st['val1'] = st['bounding_box.coordinates_str'].apply(
                    lambda x: float(x[1]))  # append the needed values as new column
                st['val3'] = st['bounding_box.coordinates_str'].apply(lambda x: float(x[3]))  # and convert them to float
                st['val0'] = st['bounding_box.coordinates_str'].apply(lambda x: float(x[0]))
                st['val4'] = st['bounding_box.coordinates_str'].apply(lambda x: float(x[4]))

                st['center_coord_X'] = (st['val0'] + st['val4']) / 2  # bounding box-center x-coordinate
                st['center_coord_Y'] = (st['val1'] + st['val3']) / 2  # bounding box-center y-coordinate
                self.raw_data = pd.concat([self.raw_data, st], axis=1)  # append the X and Y coordinates to the dataframe

                # Tokenization (usage of static method):
                self.raw_data['text_tokens'] = self.raw_data['text'].apply(lambda x: Cleaner._tokenizer(self.spacy_model, x,self.text_case))
                
                def is_land(latitude, longitude):
                    geolocator = Nominatim(user_agent="land_checker")
                    try:
                        location = geolocator.reverse(f"{latitude}, {longitude}", exactly_one=True, language="en")
                        return location.raw['address']['country']
                    except:
                        return None
                if 'country' not in self.raw_data.columns:
                    # Remove coordinates that do not belong to any country
                    self.raw_data['country'] = self.raw_data.apply(lambda row: is_land(row['center_coord_Y'], row['center_coord_X']), axis=1)

            else:
                if 'country' not in self.raw_data.columns:
                    if 'center_coord_X' in self.raw_data.columns:
                        
                        def is_land(latitude, longitude):
                            geolocator = Nominatim(user_agent="land_checker")
                            try:
                                location = geolocator.reverse(f"{latitude}, {longitude}", exactly_one=True, language="en")
                                return location.raw['address']['country']
                            except:
                                return None

                        # Remove coordinates that do not belong to any country
                        self.raw_data['country'] = self.raw_data.apply(lambda row: is_land(row['center_coord_Y'], row['center_coord_X']), axis=1)
                    
                    
            
            # Tokenization (usage of static method):
            self.raw_data['text_tokens'] = self.raw_data['text'].apply(lambda x: Cleaner._tokenizer(self.spacy_model, x,self.text_case))
            
            if self.min_post_len is not None:
                # check the length of a tweet:
                len_text = self.raw_data['text_tokens'].apply(lambda x: len(x))  # get the length of all text fields
                self.raw_data = self.raw_data[
                    len_text > self.min_post_len]  # take only texts with more than 100 characters


            
            if self.data == 'twitter':
                self.raw_data = self.raw_data.loc[:, ['created_at', 'text', 'text_tokens', 'hashtags', 'center_coord_X','center_coord_Y','country']]
            else:
                if 'center_coord_X' in self.raw_data.columns:
                    self.raw_data['created_at'] = datetime.now()
                    self.raw_data = self.raw_data.loc[:, ['created_at', 'text', 'text_tokens', 'hashtags','center_coord_X','center_coord_Y','country']]
                    
                else:
                    
                    self.raw_data['created_at'] = datetime.now()
                    self.raw_data = self.raw_data.loc[:, ['created_at', 'text', 'text_tokens', 'hashtags']]
                    
                    
           
            return self.raw_data

    @staticmethod  # using static method, see: https://realpython.com/instance-class-and-static-methods-demystified/
    def _tokenizer(nlp,text,text_case):
        # "nlp" Object is used to create documents with linguistic annotations.
        doc = nlp(text)

        # Create list of word tokens:
        # remove stop-words, non-alphabethical words, punctuation, words shorter than three characters and every
        # word that contains the sub-string 'amp'. From these words, keep only Proper Nouns, Nouns, Adjectives and Verbs
        token_list_topic_model = []
        for token in doc:
            if (token.is_stop == False) & (token.is_alpha == True) & (token.pos_ != 'PUNCT') & (len(token) > 2) & (
                    re.search('amp', str(token)) == None) & ((token.pos_ == 'PROPN') | (token.pos_ == 'NOUN')
                                                             | (token.pos_ == 'ADJ') | (token.pos_ == 'VERB')):
                
                if text_case:
                  token_list_topic_model.append(token.lemma_.lower())  # tokens for topic models
                else:
                  token_list_topic_model.append(token.lemma_)  # tokens for topic models

        return token_list_topic_model

    # _stat_func_caller = _tokenizer.__func__(spacy_model,b) #ensure callability of static method inside instance
    # method, see: https://stackoverflow.com/questions/12718187/calling-class-staticmethod-within-the-class-body


    def saving(self, save_path, data_save_name='my_cleaned_and_tokenized_data'):
        """
        Initialize the TopicModeling object.

        Args:
            data_save_name (str): Name of the cleaned and tokenized data file (default: 'my_cleaned_and_tokenized_data').

            save_path (str): Path to save the scraped data. 

        """

        assert isinstance(data_save_name, str), "data_save_name should be a string." 
        assert isinstance(save_path, str), "save_path should be a string."

        self.data_save_name = data_save_name
        # save data as pickle or csv.
        _pack_size = 100000
        parts_to_save = math.ceil(len(self.clean_data) / _pack_size)  # calculate how many parts to save (rounds up)
        upper_bound = _pack_size
        for i in range(0, parts_to_save):
            lower_bound = upper_bound - _pack_size
            file_to_save = self.clean_data.iloc[lower_bound:upper_bound, :]
            upper_bound = upper_bound + _pack_size
            file_to_save = file_to_save[file_to_save['text'] != 'nan'].reset_index(drop=True)
            if self.data == 'twitter':
              file_to_save = file_to_save.dropna().reset_index(drop=True)
              if self.file_format == 'pkl' :
                file_to_save.to_pickle(os.path.join(save_path, self.data_save_name + '_part_twitter_' + str(i + 1) + '.pkl'))
              else:
                file_to_save.to_csv(os.path.join(save_path, self.data_save_name + '_part_twitter_' + str(i + 1) + '.csv'))
            
            else:
            
              file_to_save = file_to_save.dropna().reset_index(drop=True)
              file_to_save = file_to_save[file_to_save['text_tokens'].str.len() >= 1].reset_index(drop=True)
              if self.file_format == 'pkl' :
                  file_to_save.to_pickle(os.path.join(save_path, self.data_save_name + '_part_' + str(i + 1) + '.pkl'))
              else:
                  file_to_save.to_csv(os.path.join(save_path, self.data_save_name + '_part_' + str(i + 1) + '.csv'))
        
        return file_to_save
    
