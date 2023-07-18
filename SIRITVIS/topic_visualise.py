

#or
#if you wanna use the module right away 
#import subprocess,sys
#subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==0.23.2"])

import glob
from heapq import nlargest

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import topicwizard
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
import os
import matplotlib.pyplot as plt

import pandas as pd
from IPython.core.display import display, HTML

import pyLDAvis
import pyLDAvis.lda_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings

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
        - file_path (str or DataFrame): The path to the CSV file or a DataFrame containing the data.
        - text_column (str): The name of the text column in the CSV file or DataFrame.
        """
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
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        

    

class TopicWizardvis():
    def __init__(self, data_source, num_topics=10, text_column='text'):
        self.csv_file = data_source
        self.num_topics = num_topics
        self.column_name = text_column
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


            cor = df[self.column_name].str.replace('rt', '').tolist()
            # Create topic pipeline
            pipeline = make_pipeline(
                CountVectorizer(stop_words="english", min_df=10),
                LatentDirichletAllocation(n_components=self.num_topics),
            )

            # Then fit it on the given texts
            pipeline.fit(cor)


            # A large corpus takes a looong time to compute 2D projections for so
            # so you can speed up preprocessing by disabling it alltogether.
            return topicwizard.visualize(cor, pipeline=pipeline, exclude_pages=["documents"])
                
        except FileNotFoundError:
            print("Error: File not found.")
        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty.")
        except KeyError:
            print(f"Error: The column '{self.column_name}' does not exist in the CSV file. Define text_column name of your dataset")
        except:
            return

        
        

 
