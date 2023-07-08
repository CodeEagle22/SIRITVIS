

#or
#if you wanna use the module right away 
#import subprocess,sys
#subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==0.23.2"])

import glob
from heapq import nlargest

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
import pandas as pd
import pyLDAvis
import pyLDAvis.lda_model
pyLDAvis.enable_notebook()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class PyLDAvis():
    def __init__(self, file_path, text_column='text'):
        """
        Initialize the LDAPackage class.

        Parameters:
        - file_path (str): The path to the CSV file.
        - text_column (str): The name of the text column in the CSV file.
        """
        self.file_path = file_path
        self.column_name = text_column
        self.df = None
        self.lines = None
        self.tf_vectorizer = None
        self.dtms_tf = None
        self.tfidf_vectorizer = None
        self.dtms_tfidf = None
        self.lda_tf = None
        self.vis = None
        self.visualize()

    def visualize(self):
        """
        Perform Latent Dirichlet Allocation (LDA) and prepare the visualization.

        Returns:
        - vis: The prepared visualization object.
        """
        try:
            # Read the CSV file and retrieve the specified column
            file_extension = os.path.splitext(self.file_path)[1]

            if file_extension == ".pkl":
                # Read pickle file
                data = pd.read_pickle(self.file_path).dropna().reset_index(drop=True)
            elif file_extension == ".csv":
                # Read CSV file
                data = pd.read_csv(self.file_path).dropna().reset_index(drop=True)
            else:
                print("Unsupported file format.")
              
            self.lines = data['text'].tolist()

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
            self.lda_tf = LatentDirichletAllocation(n_components=20, random_state=0)
            self.lda_tf.fit(self.dtms_tf)

            # Prepare the visualization
            self.vis = pyLDAvis.lda_model.prepare(self.lda_tf, self.dtms_tf, self.tf_vectorizer)

        except FileNotFoundError:
            print("Error: File not found.")
        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty.")
        except KeyError:
            print(f"Error: The column '{self.column_name}' does not exist in the CSV file. Define text_column name of your dataset")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        return self.vis




class TopicWizardvis():
    def __init__(self, csv_file, num_topics=10):
        self.csv_file = csv_file
        self.num_topics = num_topics
        self.visualize()
        

    def preprocess_data(self):
        try:
            df = pd.read_csv(self.csv_file).dropna()
            texts = df['text'].str.replace('rt', '').tolist()
            return texts
        except (FileNotFoundError, KeyError) as e:
            print(f"Error: {e}")
            return None

    def create_topic_pipeline(self):
        return make_pipeline(
            CountVectorizer(),
            LatentDirichletAllocation(n_components=self.num_topics),
        )

    def visualize(self):
        texts = self.preprocess_data()
        if texts is None:
            return
        
        topic_pipeline = self.create_topic_pipeline()

        try:
            topic_pipeline.fit(texts)
            return topicwizard.visualize(pipeline=topic_pipeline, corpus=texts)
        except Exception as e:
            print(f"Error: {e}")


