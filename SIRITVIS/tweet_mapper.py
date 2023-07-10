import pandas as pd
import folium

import pickle
import re
from ipywidgets import Dropdown, interact, Checkbox
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.sentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from folium import MacroElement
from ipyleaflet import Map, Marker, Popup, Icon
import math
import webbrowser
from IPython.display import display


class TweetMapper:
    def __init__(self, csv_file,model_file):

        """
        Initialize the TweetMapper object.

        Args:
            csv_file (str): The path to the CSV file.
            model_file (str or dict): The path to the model file (if str) or the model dictionary (if dict).
        """

        assert isinstance(csv_file, str), "csv_file must be a string"
        assert isinstance(model_file, (str, dict)), "model_file must be a string path or a dictionary variable"

        # Read the data into a pandas DataFrame
        self.df = pd.read_csv(csv_file)[:15000]
        self.model_file = model_file
        try:
          self.center_lat = self.df['center_coord_Y'].mean()
          self.center_lon = self.df['center_coord_X'].mean()
        except:
          print("Make sure csv_file contains coordinate columns 'center_coord_Y' and 'center_coord_X'")
        self.keyword_rankings = {}
        self.country_dropdown = None

        # Preprocess text tokens
        self.preprocess_text_tokens()

        # Create dropdowns
        self.create_dropdowns()

    def perform_sentiment_analysis(self):
        # Initialize the sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        sentiments = []

        # Perform sentiment classification for each text in the dataset
        for index, row in self.df.iterrows():
            text = row['text']
            if isinstance(text, str):
                scores = analyzer.polarity_scores(text)
                compound_score = scores['compound']
                sentiments.append(compound_score)
            else:
                sentiments.append(0)  # Set a neutral sentiment for non-string values


        # Add 'sentiment' column to the DataFrame
        self.df['sentiment'] = sentiments

        # Calculate positive, negative, and neutral tweet counts
        self.df['positive_tweet_count'] = self.df['sentiment'].apply(lambda x: 1 if x > 0 else 0)
        self.df['negative_tweet_count'] = self.df['sentiment'].apply(lambda x: 1 if x < 0 else 0)
        self.df['neutral_tweet_count'] = self.df['sentiment'].apply(lambda x: 1 if x == 0 else 0)

        return self.df

    def preprocess_text_tokens(self):
        # Remove unnecessary characters from text tokens
        self.df['text_tokens'] = self.df['text_tokens'].str.replace('\'', '')

    def merge_data(self, keyword=None):
        self.df = self.perform_sentiment_analysis()

        # Merge data by grouping on coordinates and aggregating text tokens and sentiment

        merged_df = self.df.groupby(['center_coord_X', 'center_coord_Y', 'country']).agg({'text_tokens': lambda x: ''.join([item for sublist in x for item in sublist]), 'sentiment': 'mean', 'positive_tweet_count': 'sum', 'negative_tweet_count': 'sum', 'neutral_tweet_count': 'sum'}).reset_index()

        # Check if the selected keyword is present in all records with the same 'center_coord_X' and 'center_coord_Y'
        if keyword:
            filtered_df = merged_df[merged_df['text_tokens'].apply(lambda x: keyword in x)]
            grouped_df = filtered_df.groupby(['center_coord_X', 'center_coord_Y', 'country']).size().reset_index(name='count')
            same_keyword_coords = set(grouped_df[grouped_df['count'] == grouped_df['count'].max()][['center_coord_X', 'center_coord_Y','country']].itertuples(index=False))

            # Exclude sentiment mean aggregation if the keyword is present in all records with the same coordinates
            merged_df.loc[merged_df.apply(lambda row: (row['center_coord_X'], row['center_coord_Y'],row['country']) in same_keyword_coords, axis=1), 'sentiment'] = None

        return merged_df


    def filter_dataset(self):
        merge_df = self.merge_data()

        # Load the model and get merged keywords
        if type(self.model_file)==dict:
              model = self.model_file
              merged_keywords = [item for sublist in model['topics'] for item in sublist]
              self.keyword_rankings = {keyword: rank + 1 for rank, keyword in enumerate(merged_keywords)}
        else:
          with open(self.model_file, 'rb') as file:
              model = pickle.load(file)
              merged_keywords = [item for sublist in model['topics'] for item in sublist]
              self.keyword_rankings = {keyword: rank + 1 for rank, keyword in enumerate(merged_keywords)}

        filtered_dataset = merge_df.copy()

        # Filter the dataset based on keyword rankings and length of tokens
        filtered_dataset['text_tokens'] = filtered_dataset['text_tokens'].apply(lambda x: re.findall(r'\w+', x)).apply(lambda x: list(set(x))).apply(lambda x: [token for token in x if len(token) >= 3]).apply(lambda x: sorted(set(x), key=lambda token: self.keyword_rankings.get(token, float('inf')))[:10])

        filtered_dataset = filtered_dataset[filtered_dataset['text_tokens'].apply(lambda x: any(token in self.keyword_rankings for token in x))]

        return filtered_dataset, merged_keywords

    def create_dropdowns(self):
        merged_df, merged_keyword = self.filter_dataset()
        keywords = merged_keyword.copy()[:10]
        options = [' '] + keywords

        # Create a keyword dropdown widget
        keyword_dropdown = Dropdown(options=options, description='Select Topic To Visualize: <br>', value=None)
        keyword_dropdown.observe(self.on_dropdown_change, names='value')
        self.keyword_dropdown = keyword_dropdown

        # Create a country dropdown widget
        country_options = [' '] + sorted(list(self.df['country'].unique()))  # Sort the country list in ascending order
        country_dropdown = Dropdown(options=country_options, description='Select Country: <br>', value=None)
        country_dropdown.observe(self.on_country_dropdown_change, names='value')
        self.country_dropdown = country_dropdown

        # Set initial options for the keyword dropdown based on the first selected country
        if country_dropdown.value:
            initial_country = country_dropdown.value
            initial_keywords = self.get_keywords_by_country(initial_country)
            initial_keyword_options = [' '] + initial_keywords
            self.keyword_dropdown.options = initial_keyword_options

        # Interact with the add_markers method based on dropdown values
        interact(self.add_markers, keyword=self.keyword_dropdown, country=self.country_dropdown)




    def on_dropdown_change(self, change):
        # Reset keyword dropdown value if ' ' is selected
        if change['new'] == ' ':
            self.keyword_dropdown.value = None

    def on_country_dropdown_change(self, change):
        # Reset country dropdown value if ' ' is selected
        if change['new'] == ' ':
            self.country_dropdown.value = None
            self.keyword_dropdown.options = [' '] + self.filter_dataset()[1]
        else:
            selected_country = change['new']
            keywords = self.get_keywords_by_country(selected_country)
            options = [' '] + keywords
            self.keyword_dropdown.options = options

    def get_keywords_by_country(self, country):
        filtered_df = self.df[self.df['country'] == country]
        keywords = filtered_df['text_tokens'].apply(lambda x: re.findall(r'\w+', x)).apply(lambda x: list(set(x))).apply(lambda x: [token for token in x if len(token) >= 3]).apply(lambda x: sorted(set(x), key=lambda token: self.keyword_rankings.get(token, float('inf'))))
        flattened_keywords = [keyword for sublist in keywords for keyword in sublist]
        unique_keywords = list(set(flattened_keywords))
        sorted_keywords = sorted(unique_keywords, key=lambda token: self.keyword_rankings.get(token, float('inf')))[:10]
        return sorted_keywords


    def add_markers(self, keyword=None, country=None, enable_sentiment=True, enable_tweet_count=False):
        # Clear the map
        self.map = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=2)

        merged_df, merged_keywords = self.filter_dataset()
        filtered_df = merged_df.copy()

        if keyword:
            self.df = self.df[self.df['text_tokens'].apply(lambda x: keyword in x)]
            self.df['text_tokens'] = self.df['text_tokens'].str.replace('\'', '')

        if country:
            self.df = self.df[self.df['country'] == country]


        filtered_df = self.df.groupby(['center_coord_X', 'center_coord_Y', 'country']).agg({'text_tokens': lambda x: ''.join([item for sublist in x for item in sublist]), 'sentiment': 'mean', 'positive_tweet_count': 'sum', 'negative_tweet_count': 'sum', 'neutral_tweet_count': 'sum'}).reset_index()

        for index, row in filtered_df.iterrows():
            lat = row['center_coord_Y']
            lon = row['center_coord_X']
            input_string = row['text_tokens'].strip('[]')
            output_list = input_string.split(',')
            row['text_tokens'] = [item.strip() for item in output_list]
            tokens = list(set(row['text_tokens']))[:10]
            sentiment = row['sentiment']
            country_name = row['country']


            popup_text = ""

            if country_name:
                popup_text += f"<b>{country_name}</b> <br><br>"

            count = ""
            if enable_tweet_count:

                if keyword and country:
                    pos_count = row['positive_tweet_count']
                    neg_count = row['negative_tweet_count']
                    neu_count = row['neutral_tweet_count']
                    count = f"<span style='color: blue'>Total tweets: {pos_count + neg_count + neu_count}</span><br><span style='color: green'>Positive tweets: {pos_count}</span><br><span style='color: red'>Negative tweets: {neg_count}</span><br><span style='color: gray'>Neutral tweets: {neu_count}</span>"

                elif country:
                    pos_count = row['positive_tweet_count']
                    neg_count = row['negative_tweet_count']
                    neu_count = row['neutral_tweet_count']
                    original_filtered_df = self.df[self.df['country'] == country]
                    tweet_count = len(original_filtered_df)
                    positive_count = len(original_filtered_df[original_filtered_df['sentiment'] > 0])
                    negative_count = len(original_filtered_df[original_filtered_df['sentiment'] < 0])
                    neutral_count = len(original_filtered_df[original_filtered_df['sentiment'] == 0])
                    count = f"<span style='color: blue'>Total tweets: {pos_count + neg_count + neu_count}</span>  <b>({tweet_count})</b><br><span style='color: green'>Positive tweets: {pos_count}</span>  <b>({positive_count})</b><br><span style='color: red'>Negative tweets: {neg_count}</span>  <b>({negative_count})</b><br><span style='color: gray'>Neutral tweets: {neu_count}</span>  <b>({neutral_count})</b>"

                elif keyword:
                    pos_count = row['positive_tweet_count']
                    neg_count = row['negative_tweet_count']
                    neu_count = row['neutral_tweet_count']
                    original_filtered_df = self.df[self.df['text_tokens'].apply(lambda x: keyword in x)]
                    tweet_count = len(original_filtered_df)
                    positive_count = len(original_filtered_df[original_filtered_df['sentiment'] > 0])
                    negative_count = len(original_filtered_df[original_filtered_df['sentiment'] < 0])
                    neutral_count = len(original_filtered_df[original_filtered_df['sentiment'] == 0])
                    count = f"<span style='color: blue'>Total tweets: {pos_count + neg_count + neu_count}</span><br><span style='color: green'>Positive tweets: {pos_count}</span><br><span style='color: red'>Negative tweets: {neg_count}</span><br><span style='color: gray'>Neutral tweets: {neu_count}</span>"

                else:
                    pos_count = row['positive_tweet_count']
                    neg_count = row['negative_tweet_count']
                    neu_count = row['neutral_tweet_count']
                    count = f"<span style='color: blue'>Total tweets: {pos_count + neg_count + neu_count}</span><br><span style='color: green'>Positive tweets: {pos_count}</span><br><span style='color: red'>Negative tweets: {neg_count}</span><br><span style='color: gray'>Neutral tweets: {neu_count}</span>"

            else:

                popup_text += "<br>".join([f"<b>{rank} {token.capitalize()}</b>" if token == keyword else f"<b>{rank}</b> {token.capitalize()}" for rank, token in enumerate(tokens, start=1)])

            popup_width = max(len(max(popup_text.split("<br>"), key=len)), 200)  # Adjust popup width

            popup_text += count if enable_tweet_count else ""



            # Determine the color of the marker based on sentiment if enabled, otherwise use the default color
            if enable_sentiment:
                if sentiment is not None:
                    if sentiment > 0:
                        color = 'green'
                    elif sentiment < 0:
                        color = 'red'
                    else:
                        color = 'blue'
                else:
                    color = 'blue'
            else:
                color = 'blue'  # Default color when sentiment-based coloring is disabled

            marker = folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_text, max_width=popup_width),
                icon=folium.Icon(icon='info-sign', color=color)
            )
            marker.add_to(self.map)


        # Display the map
        display(self.map)




