# Copyright (c) [year] [your name]
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from ipywidgets import Button, Output
import pandas as pd
import folium
from ipywidgets import Dropdown, interact, Checkbox
import re
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from IPython.display import display, HTML
from branca.element import Element
import folium
import math
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

# Adjust log level to suppress log messages
import logging
logging.getLogger().setLevel(logging.ERROR)

class TopicMapper:
    def __init__(self, data_source, model_source):
        """
        Initialize the TopicMapper object.

        Args:
            data_source (str,pd.DataFrame): The path to the CSV file or clean_data variable.
            model_file (str or dict): The path to the model file (if str) or the model dictionary (if dict).
        """

        assert isinstance(data_source, (str,pd.DataFrame)), "csv_file must be a string"
        assert isinstance(model_source, (str, dict)), "model_file must be a string path or a dictionary variable"

        # Read the data into a pandas DataFrame

        if isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
        else:
            self.df = data_source

        self.model_file = model_source
        try:
          self.center_lat = self.df['center_coord_Y'].mean()
          self.center_lon = self.df['center_coord_X'].mean()
        except:
          print("Make sure csv_file contains coordinate columns 'center_coord_Y' and 'center_coord_X'")
        self.keyword_rankings = {}
        self.country_dropdown = None
        # Create an output widget to display download status
        self.output_widget = Output()
        
        # Create an export button
        self.export_button = Button(description="Export Map as HTML")
        self.export_button.on_click(self.export_map)

     

        # Create dropdowns
        self.create_dropdowns()

    def perform_sentiment_analysis(self):
        # Initialize the sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        sentiments = []

        # Perform sentiment classification for each text in the dataset
        for index, row in self.df.iterrows():  # Iterate over DataFrame rows
            text = row['text']  # Access 'text' column using row indexing
            if isinstance(text, str):
                scores = analyzer.polarity_scores(text)
                compound_score = scores['compound']
                sentiments.append(compound_score)
            else:
                sentiments.append(0)  # Set a neutral sentiment for non-string values

        # Add 'sentiment' column to the DataFrame
        self.df['sentiment'] = sentiments

        # Calculate positive, negative, and neutral post counts
        self.df['positive_post_count'] = self.df['sentiment'].apply(lambda x: 1 if x > 0 else 0)
        self.df['negative_post_count'] = self.df['sentiment'].apply(lambda x: 1 if x < 0 else 0)
        self.df['neutral_post_count'] = self.df['sentiment'].apply(lambda x: 1 if x == 0 else 0)
        self.df['text_tokens'] = self.df['text_tokens'].astype(str)
        self.df['text_tokens'] = self.df['text_tokens'].str.replace("\'", "")

        return self.df
    
        
    def export_map(self, _):
        # Get the selected keyword and country from the dropdowns
        selected_keyword = self.keyword_dropdown.value if self.keyword_dropdown.value else "All"
        selected_country = self.country_dropdown.value if self.country_dropdown.value else "All"

        # Generate a filename based on the selected keyword and country
        export_filename = f"map_{selected_country}_{selected_keyword}.html"
        export_filename = export_filename.replace(" ", "_")  # Replace spaces with underscores

        # Get the map as an HTML string
        map_html = self.map.get_root().render()

        # Save the map HTML to the generated filename
        with open(export_filename, "w", encoding="utf-8") as html_file:
            html_file.write(map_html)

        # Display a message in the output widget
        with self.output_widget:
            print(f"Map exported as '{export_filename}'")


    def filter_dataset(self,keyword=None,country=None):
        df = self.perform_sentiment_analysis()
        # Load the model and get merged keywords
         
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

        # Calculate the mean topic-document matrix
        n = math.floor(len(model['topic-document-matrix'][0]) / len(model['topics'][0]))
        # Check if n is 0 to avoid division by zero
        if n == 0:
            # Set mean_list to an empty list since we can't divide by 0
            mean_list = []
        else:
            mean_list = [[sum(sublist[i:i+n]) / n for i in range(0, len(sublist), n)] for sublist in model['topic-document-matrix'].tolist()]

        # Flatten the mean_list to get two_list
        two_list = [item for sublist in mean_list for item in sublist[:len(model['topics'][0])]]

        total_count = sum(two_list)

        # Create a DataFrame with the mean topic-document matrix values and percentages
        percentage_list = [(item / total_count) * 100 for item in two_list]
        dfr = pd.DataFrame({'tdm': two_list, 'values': merged_keywords, 'percentage': percentage_list})

        # Calculate the mean topic-document matrix by values
        mean_tdm = dfr.groupby('values')['tdm'].mean().reset_index().sort_values('tdm', ascending=False).reset_index(drop=True)

        
        filtered_dataset = df.copy()

        # Convert the 'text_tokens' column to string type if necessary
        filtered_dataset['text_tokens'] = filtered_dataset['text_tokens'].astype(str)

        # Filter the dataset based on keyword rankings and length of tokens
        filtered_dataset['text_tokens'] = (filtered_dataset['text_tokens']
                                          .apply(lambda x: [token for token in re.findall(r'\w+', x) if len(token) >= 3])
                                          .apply(lambda x: sorted(set(x), key=lambda token: self.keyword_rankings.get(token, float('inf'))))
                                          )

        filtered_dataset = filtered_dataset[filtered_dataset['text_tokens'].apply(lambda x: any(token in self.keyword_rankings for token in x))]

        if keyword:
          filtered_dataset = filtered_dataset[filtered_dataset['text_tokens'].apply(lambda x: keyword in x)]


        filtered_dataset['text_token_dict'] = (filtered_dataset['text_tokens']
                                          .apply(lambda x: {value: mean_tdm.loc[mean_tdm['values'] == value, 'tdm'].values[0] for value in x if value in mean_tdm['values'].values})
                                          .apply(lambda x: {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)})
                                        )
        merged_keyword = mean_tdm['values'].to_list()
        return filtered_dataset, merged_keyword

    def merge_data(self,dataset):
        merge_df = dataset
        
        
        # Merge data by grouping on coordinates and aggregating text tokens and sentiment
        merged_df = merge_df.groupby(['center_coord_X', 'center_coord_Y','country']).agg(
              {
                  'text_token_dict': lambda x: dict(sorted({k: v for item in x for k, v in item.items()}.items(), key=lambda item: item[1], reverse=True)),
                  'text_tokens': lambda x: ''.join([item for sublist in x for item in sublist]),
                  'sentiment': 'mean',
                  'positive_post_count': 'sum',
                  'negative_post_count': 'sum',
                  'neutral_post_count': 'sum'
              }
          ).reset_index()


         

        
        return merged_df


    def create_dropdowns(self):
        merged_df, merged_keyword = self.filter_dataset()
        keywords = merged_keyword.copy()[:50]
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


        # Add the export button to the widget
        interact(self.add_markers, keyword=self.keyword_dropdown, country=self.country_dropdown)
        
        # Display the export button
        display(self.export_button)
        display(self.output_widget)


    def on_dropdown_change(self, change):
        # Reset keyword dropdown value if ' ' is selected
        if change['new'] == ' ':
            self.keyword_dropdown.value = None

    def on_country_dropdown_change(self, change):
        # Reset country dropdown value if ' ' is selected
        merged_df, merged_keyword = self.filter_dataset()
        if change['new'] == ' ':
            self.country_dropdown.value = None
            self.keyword_dropdown.options = [' '] + merged_keyword.copy()[:50]
        else:
            selected_country = change['new']
            keywords = self.get_keywords_by_country(selected_country)
            options = [' '] + merged_keyword.copy()[:50]
            self.keyword_dropdown.options = options

    def get_keywords_by_country(self, country):
        filtered_df = self.df[self.df['country'] == country]
        keywords = filtered_df['text_tokens'].apply(lambda x: re.findall(r'\w+', x)).apply(lambda x: list(set(x))).apply(lambda x: [token for token in x if len(token) >= 3]).apply(lambda x: sorted(set(x), key=lambda token: self.keyword_rankings.get(token, float('inf'))))
        flattened_keywords = [keyword for sublist in keywords for keyword in sublist]
        unique_keywords = list(set(flattened_keywords))
        sorted_keywords = sorted(unique_keywords, key=lambda token: self.keyword_rankings.get(token, float('inf')))[:10]
        return sorted_keywords


    def add_markers(self, keyword=None, country=None, enable_sentiment=True, enable_post_count=False):
        # Clear the map
        self.map = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=2,max_zoom=15, min_zoom=2)
        
        if keyword:
            filtered_dataset, merged_keywords = self.filter_dataset(keyword=keyword)
        elif keyword and country:
            filtered_dataset, merged_keywords = self.filter_dataset(keyword=keyword)
            filtered_dataset = filtered_dataset[filtered_dataset['country'] == country]
        else:
            filtered_dataset, merged_keywords = self.filter_dataset()
        merged_df = self.merge_data(filtered_dataset)
        filtered_df = merged_df.copy()

        if country:
            filtered_df = filtered_df[filtered_df['country'] == country]

        # Perform sentiment analysis on the original filtered DataFrame
        analyzer = SentimentIntensityAnalyzer()
        sentiments = []

        for index, row in filtered_df.iterrows():
            lat = row['center_coord_Y']
            lon = row['center_coord_X']
            tokens = list(row['text_token_dict'].keys())
            frequency = list(row['text_token_dict'].values())
            total = sum(row['text_token_dict'].values())
            if row['positive_post_count'] > row['negative_post_count'] and row['positive_post_count'] > row['neutral_post_count']:
                priority = 1
            elif row['negative_post_count'] > row['positive_post_count']and row['negative_post_count'] > row['neutral_post_count']:
                priority = 2
            else:
                priority = 3
            sentiment = row['sentiment']
            country_name = row['country']

            popup_text = ""

            if country_name:
                popup_text += f"<b>{country_name}</b> <br><br>"

            count = ""
            if enable_post_count:
                if keyword and country:
                    summed_counts = filtered_df.groupby('country').agg({
                        'positive_post_count': 'sum',
                        'negative_post_count': 'sum',
                        'neutral_post_count': 'sum'
                    }).reset_index()
                    post_count = int(summed_counts['positive_post_count'] + summed_counts['negative_post_count'] + summed_counts['neutral_post_count'])
                    positive_count = int(summed_counts['positive_post_count'])
                    negative_count = int(summed_counts['negative_post_count'])
                    neutral_count = int(summed_counts['neutral_post_count'])
                    pos_count = row['positive_post_count']
                    neg_count = row['negative_post_count']
                    neu_count = row['neutral_post_count']
                    count = f"<span style='color: blue'>Total posts: {pos_count + neg_count + neu_count}</span>  <b>({post_count})</b><br><span style='color: green'>Positive posts: {pos_count}</span>  <b>({positive_count})</b><br><span style='color: red'>Negative posts: {neg_count}</span>  <b>({negative_count})</b><br><span style='color: gray'>Neutral posts: {neu_count}</span>  <b>({neutral_count})</b>"
                elif keyword:
                    pos_count = row['positive_post_count']
                    neg_count = row['negative_post_count']
                    neu_count = row['neutral_post_count']
                    count = f"<span style='color: blue'>Total posts: {pos_count + neg_count + neu_count}</span><br><span style='color: green'>Positive posts: {pos_count}</span><br><span style='color: red'>Negative posts: {neg_count}</span><br><span style='color: gray'>Neutral posts: {neu_count}</span>"
                elif country:
                    summed_counts = filtered_df.groupby('country').agg({
                        'positive_post_count': 'sum',
                        'negative_post_count': 'sum',
                        'neutral_post_count': 'sum'
                    }).reset_index()
                    post_count = int(summed_counts['positive_post_count'] + summed_counts['negative_post_count'] + summed_counts['neutral_post_count'])
                    positive_count = int(summed_counts['positive_post_count'])
                    negative_count = int(summed_counts['negative_post_count'])
                    neutral_count = int(summed_counts['neutral_post_count'])

                    pos_count = row['positive_post_count']
                    neg_count = row['negative_post_count']
                    neu_count = row['neutral_post_count']
                    count = f"<span style='color: blue'>Total posts: {pos_count + neg_count + neu_count}</span>  <b>({post_count})</b><br><span style='color: green'>Positive posts: {pos_count}</span>  <b>({positive_count})</b><br><span style='color: red'>Negative posts: {neg_count}</span>  <b>({negative_count})</b><br><span style='color: gray'>Neutral posts: {neu_count}</span>  <b>({neutral_count})</b>"
                else:
                    pos_count = row['positive_post_count']
                    neg_count = row['negative_post_count']
                    neu_count = row['neutral_post_count']
                    count = f"<span style='color: blue'>Total posts: {pos_count + neg_count + neu_count}</span><br><span style='color: green'>Positive posts: {pos_count}</span><br><span style='color: red'>Negative posts: {neg_count}</span><br><span style='color: gray'>Neutral posts: {neu_count}</span>"
            else:
                popup_text += "<br>".join([
                    f"<b>{rank} {token.capitalize()} {round((fre / sum(frequency)) * 100, 2)}%</b>"
                    if token == keyword else f"<b>{rank}</b> {token.capitalize()} {round((fre / sum(frequency)) * 100, 2)}%"
                    for rank, (token, fre) in enumerate(zip(tokens[:10], frequency[:10]), start=1)
                ])

                if len(tokens) > 10:
                    popup_text += f"<br><b>Other:</b> {round((sum(frequency[10:]) / sum(frequency)) * 100, 2)}%"

            popup_width = max(len(max(popup_text.split("<br>"), key=len)), 200)  # Adjust popup width

            popup_text += count if enable_post_count else ""

            # Determine the color of the marker based on sentiment if enabled, otherwise use the default color
            if enable_sentiment:
                if sentiment is not None:
                    if priority == 1:
                        color = 'green'
                    elif priority == 2:
                        color = 'red'
                    elif priority == 3:
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
        display(HTML(self.map._repr_html_()))
