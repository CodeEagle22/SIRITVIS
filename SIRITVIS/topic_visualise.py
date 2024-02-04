# Copyright (c) [year] [your name]
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import glob
from heapq import nlargest
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display import display, HTML
import pyLDAvis
from pyLDAvis.lda_model import LDA
import warnings
from wordcloud import WordCloud


# Suppress warnings
warnings.filterwarnings("ignore")

# Adjust log level to suppress log messages
import logging
logging.getLogger().setLevel(logging.ERROR)

class PyLDAvis():
    def __init__(self, data_source,num_topics=10, text_column='text'):
        """
        Initialize the PyLDAvis class.

        Parameters:
        - data_source (str or DataFrame): The path to the CSV file or a DataFrame containing the data.
        - text_column (str): The name of the text column in the CSV file or DataFrame.
        """
        assert isinstance(data_source, (str,pd.DataFrame)), "data_source should be a string path or preprocessed dataset variable"
        assert text_column is None or isinstance(text_column, str), "text_column must be a str"
        self.file_path = data_source
        self.column_name = text_column
        self.num_topics = num_topics
        self.lines = None
        self.tf_vectorizer = None
        self.dtms_tf = None
        self.tfidf_vectorizer = None
        self.dtms_tfidf = None
        self.lda_tf = None
        self.vis = None

    def visualize(self):
        """
        Perform Latent Dirichlet Allocation (LDA) and prepare the visualization.

        Returns:
        - vis: The prepared visualization object.
        """
        try:

            print('The visualisation is based on Latent Dirichlet Allocation (LDA) model.')
            # Read the CSV file or use the provided DataFrame
            if isinstance(self.file_path, pd.DataFrame):
                data = self.file_path
            elif isinstance(self.file_path, str):
                file_extension = os.path.splitext(self.file_path)[1]
                if file_extension == ".pkl":
                    # Read pickle file
                    data = pd.read_pickle(self.file_path).dropna().reset_index(drop=True)
                elif file_extension == ".csv":
                    # Read CSV file
                    data = pd.read_csv(self.file_path).dropna().reset_index(drop=True)
                else:
                    print("Unsupported file format.")
                    return None
            else:
                print("Invalid data type. Please provide either a DataFrame or a file path.")
                return None

            if self.column_name not in data.columns:
                print(f"Error: The column '{self.column_name}' does not exist in the data.")
                return None

            self.lines = data[self.column_name].tolist()

            # Create the TF vectorizer
            self.tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                                stop_words='english',
                                                lowercase=True,
                                                token_pattern=r'\b[a-zA-Z]{3,}\b',
                                                max_df=0.5,
                                                min_df=10)
            self.dtms_tf = self.tf_vectorizer.fit_transform(self.lines)

            # Create the TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(**self.tf_vectorizer.get_params())
            self.dtms_tfidf = self.tfidf_vectorizer.fit_transform(self.lines)

            # Perform Latent Dirichlet Allocation
            self.lda_tf = LatentDirichletAllocation(n_components=self.num_topics, random_state=0)
            self.lda_tf.fit(self.dtms_tf)

            # Prepare the visualization
            
            
            self.vis = pyLDAvis.lda_model.prepare(self.lda_tf, self.dtms_tf, self.tf_vectorizer)
            pyLDAvis.enable_notebook()
            return pyLDAvis.display(self.vis)
            
        
        except FileNotFoundError:
            print("Error: File not found.")
        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty.")
        except ValueError as ve:
            if "max_df corresponds to fewer documents than min_df" in str(ve):
                print(f"The dataset size is very small: {str(ve)}")
            else:
                print(f"An error occurred: {str(ve)}")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")

        

    

class Wordcloud():
    def __init__(self, data_source, text_column='text', save_image=False):
        """
        Initialize the Word_Cloud class.

        Parameters:
        - data_source (str or DataFrame): The path to the CSV file or a DataFrame containing the data.
        - text_column (str): The name of the text column in the CSV file or DataFrame.
        - save_image (bool): Flag indicating whether to save the word cloud image.
        """
        assert isinstance(data_source, (str,pd.DataFrame)), "data_source should be a string path or preprocessed dataset variable"
        assert text_column is None or isinstance(text_column, str), "text_column must be a str"
        assert isinstance(save_image, bool), "save_image must be a boolean"

        self.csv_file = data_source
        self.column_name = text_column
        self.save_image = save_image
        self.vis = None
        self.cloud = None
        self.word = None
        
        
    def visualize(self):
        try:
            # Read the CSV file and retrieve the specified column
            if isinstance(self.csv_file, str):
                file_extension = os.path.splitext(self.csv_file)[1]

                if file_extension == ".pkl":
                    # Read pickle file
                    df = pd.read_pickle(self.csv_file).dropna().reset_index(drop=True)
                elif file_extension == ".csv":
                    # Read CSV file
                    df = pd.read_csv(self.csv_file).dropna().reset_index(drop=True)
                else:
                    print("Unsupported file format.")
                    return None
            elif isinstance(self.csv_file, pd.DataFrame):
                df = self.csv_file.dropna().reset_index(drop=True)
            else:
                print("Unsupported data type.")
                return None

            cor = df[self.column_name].str.replace(r'\b\w{1,2}\b', '').tolist()

            # Extract the text from the DataFrame column
            input_text = ' '.join(cor)

            # Generate the word cloud with a custom colormap
            wordcloud = WordCloud(width=2000, height=1000, background_color='white', colormap='copper').generate(input_text)

            # Create an interactive plot
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')  # Turn off the axis

            # Add interaction features
            def zoom(event):
                current_xlim = plt.gca().get_xlim()
                current_ylim = plt.gca().get_ylim()
                base_scale = 1.1
                
                if event.button == 'up':
                    scale_factor = 1 / base_scale
                elif event.button == 'down':
                    scale_factor = base_scale
                else:
                    return

                new_xlim = [x * scale_factor for x in current_xlim]
                new_ylim = [y * scale_factor for y in current_ylim]

                plt.gca().set_xlim(new_xlim)
                plt.gca().set_ylim(new_ylim)
                plt.draw()

            plt.connect('scroll_event', zoom)

            # Save the word cloud image if save_image is True
            if self.save_image:
                plt.savefig('wordcloud.png')
                
            # Display the plot
            plt.show()
                
        except FileNotFoundError:
            print("Error: File not found.")
        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty.")
        except KeyError:
            print(f"Error: The column '{self.column_name}' does not exist in the CSV file. Define text_column name of your dataset")
        except Exception as e:
            print("An error occurred:", str(e))
        
        

 
