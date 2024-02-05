# Copyright (c) [year] [your name]
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pandas as pd
import string
from nltk.corpus import stopwords
from octis.models.NeuralLDA import NeuralLDA
from octis.models.ProdLDA import ProdLDA
from octis.models.LDA import LDA
from octis.models.CTM import CTM
from octis.evaluation_metrics.classification_metrics import AccuracyScore
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.similarity_metrics import PairwiseJaccardSimilarity, InvertedRBO
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.dataset.dataset import Dataset
from octis.preprocessing.preprocessing import Preprocessing
import pickle
import string
from typing import List, Union
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map  # or thread_map
from tqdm import tqdm
from pathlib import Path
from octis.dataset.dataset import Dataset
from collections import Counter
import nltk
import math
nltk.download('stopwords')
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Adjust log level to suppress log messages
import logging
logging.getLogger().setLevel(logging.ERROR)

"""
Maps the language to its corresponding spacy model
"""
spacy_model_mapping = {
    'chinese': 'zh_core_web_sm', 'danish': 'nl_core_news_sm',
    'dutch': 'nl_core_news_sm', 'english': 'en_core_web_sm',
    'french': 'fr_core_news_sm', 'german': 'de_core_news_sm',
    'greek': 'el_core_news_sm', 'italian': 'it_core_news_sm',
    'japanese': 'ja_core_news_sm', 'lithuanian': 'lt_core_news_sm',
    'norwegian': 'nb_core_news_sm', 'polish': 'pl_core_news_sm',
    'portuguese': 'pt_core_news_sm', 'romanian': 'ro_core_news_sm',
    'russian': 'ru_core_news_sm', 'spanish': 'es_core_news_sm'}


class Preprocessing:
    def __init__(
        self, lowercase: bool = True, vocabulary: List[str] = None,
        max_features: int = None, min_df: float = 0.0, max_df: float = 1.0,
        remove_punctuation: bool = True, punctuation: str = string.punctuation,
        remove_numbers: bool = True, lemmatize: bool = True,
        stopword_list: Union[str, List[str]] = None, min_chars: int = 1,
        min_words_docs: int = 0, language: str = 'english', split: bool = True, num_processes: int = None,
        save_original_indexes=True, remove_stopwords_spacy: bool = True):
        """
        init Preprocessing

        :param lowercase: if true, words in documents are reduced to
            lowercase (default: true)
        :type lowercase: boolean
        :param vocabulary: the vocabulary of the corpus to preprocess
            (default: None)
        :type vocabulary: list
        :param max_features: maximum number of words that the vocabulary must
            contain. The less frequent words will be removed. If it's not None,
            then max_df and min_df are ignored (default: None)
        :type max_features: int
        :param min_df: words below this minumum document frequency will be
            removed (default: 0.0)
        :type min_df: float
        :param max_df: words above this maximum document frequency will be
            removed (default: 1.0)
        :type max_df: float
        :param remove_punctuation: if true, punctuation will be removed
            (default: true)
        :type remove_punctuation: bool
        :param punctuation: string containing all the punctuation chars that
            need to be removed (default:
        string.punctuation)
        :type punctuation: str
        :param remove_numbers: if true, numbers will be removed
        :type remove_numbers: bool
        :param remove_stopwords_spacy: bool , if true use spacy to remove
            stopwords (default: true)
        :param lemmatize: if true, words will be lemmatized using a spacy model
            according to the language that has been set (default: true)
        :type lemmatize: bool
        :param stopword_list: if a list of strings is passed, the strings will
            be removed from the texts. Otherwise, if a str is passed, it
            represents the language of the stopwords that need to be removed.
            The stopwords are spacy's stopwords (default: None)
        :type stopword_list: str or list of str
        :param min_chars: mininum number of characters that a token should have
            (default: 1)
        :type min_chars: int
        :param min_words_docs: minimun number of words that a document should
            contain (default: 0)
        :type min_words_docs: int
        :param language: language of the documents. It needs to be set for the
            lemmatizer (default: english)
        :type language: str
        :param split: if true, the corpus will be split in train (85%),
            testing (7.5%) and validation (7.5%) set (default: true)
        :type split: bool
        
        :param num_processes: number of processes to run the preprocessing
        :type num_processes: int
        :param save_original_indexes: if true, it keeps track of the original
            indexes of the documents
        """
        self.vocabulary = vocabulary
        self.lowercase = lowercase
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.remove_punctuation = remove_punctuation
        self.punctuation = punctuation
        self.lemmatize = lemmatize
        self.language = language
        self.num_processes = num_processes
        self.remove_numbers = remove_numbers
        self.save_original_indexes = save_original_indexes

        if self.lemmatize:
            lang = spacy_model_mapping[self.language]
            try:
                self.spacy_model = spacy.load(lang)
            except IOError:
                raise IOError("Can't find model " + lang + ". Check the data directory or download it using the "
                                                           "following command:\npython -m spacy download " + lang)
        self.split = split
        

        self.remove_stopwords_spacy = remove_stopwords_spacy

        stopwords = []
        # if stopwords is None then stopwords are not removed
        if stopword_list is None:
            self.remove_stopwords_spacy = False
        else:
            # if custom list is specified, then we do not use spacy stopwords
            if type(stopword_list) == list:
                stopwords = set(stopword_list)
                self.remove_stopwords_spacy = False
            elif self.remove_stopwords_spacy:
                assert stopword_list == language
            else:
                # if remove_stopwords_spacy is false, then use MALLET English stopwords
                if 'english' in stopword_list:
                    stop_word_path = Path(__file__).parent.joinpath('stopwords', 'english.txt')
                    with open(stop_word_path) as fr:
                        stopwords = [line.strip() for line in fr.readlines()]
                        assert stopword_list == language

        self.stopwords = stopwords
        self.min_chars = min_chars
        self.min_doc_words = min_words_docs
        self.preprocessing_steps = []

    def preprocess_dataset(self, documents_path, labels_path=None, multilabel=False):
        """
        preprocess the input dataset

        :param documents_path: path to the documents file. Each row of the file represents a document
        :type documents_path: str
        :param labels_path: path to the documents file. Each row of the file represents a label. Its index corresponds
        to the index of the documents file (default: None)
        :type labels_path: str
        :param multilabel: if true, a document is supposed to have more than one label (labels are split by whitespace)
        :type multilabel: bool

        :return octis.dataset.dataset.Dataset
        """
        docs = [line.strip() for line in open(documents_path, 'r').readlines()]
        if self.num_processes is not None:
            # with Pool(self.num_processes) as p:
            #    docs = p.map(self.simple_preprocessing_steps, docs)
            chunksize = max(1, len(docs) // (self.num_processes * 20))
            docs_list = process_map(self.simple_preprocessing_steps, docs, max_workers=self.num_processes, chunksize=chunksize)
        else:
            docs = list(map(self.simple_preprocessing_steps, tqdm(docs)))
        if self.lowercase:
            self.preprocessing_steps.append("lowercase")
        if self.remove_punctuation:
            self.preprocessing_steps.append('remove_punctuation')
        if self.lemmatize:
            self.preprocessing_steps.append('lemmatize')

        vocabulary = self.filter_words(docs)
        print("created vocab")
        print(len(vocabulary))
        final_docs, final_labels, document_indexes = [], [], []
        if labels_path is not None:
            if multilabel:
                labels = [
                    line.strip().split()
                    for line in open(labels_path, 'r').readlines()]
            else:
                labels = [
                    line.strip()
                    for line in open(labels_path, 'r').readlines()]

            vocab = set(vocabulary)
            for i, doc, label in zip(range(len(docs)), docs, labels):
                new_doc = [w for w in doc.split() if w in vocab]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    final_labels.append(label)
                    document_indexes.append(i)

            labels_to_remove = set([k for k, v in dict(
                Counter(final_labels)).items() if v <= 3])
            if len(labels_to_remove) > 0:
                docs = final_docs
                labels = final_labels
                document_indexes, final_labels, final_docs = [], [], []
                for i, doc, label in zip(range(len(docs)), docs, labels):
                    if label not in labels_to_remove:
                        final_docs.append(doc)
                        final_labels.append(label)
                        document_indexes.append(i)
        else:
            vocab = set(vocabulary)
            for i, doc in enumerate(docs):
                new_doc = [w for w in doc.split() if w in vocab]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    document_indexes.append(i)

        self.preprocessing_steps.append('filter documents with less than ' + str(self.min_doc_words) + " words")
        
        metadata = {"total_documents": len(docs), "vocabulary_length": len(vocabulary),
                    "preprocessing-info": self.preprocessing_steps
                    # ,"labels": list(set(final_labels)), "total_labels": len(set(final_labels))
                    }
        if self.split:
            if len(final_labels) > 0:
                train, test, y_train, y_test = train_test_split(
                    range(len(final_docs)), final_labels, test_size=0.15, random_state=1, shuffle=True)#stratify=final_labels)

                train, validation = train_test_split(train, test_size=3 / 17, random_state=1, shuffle=True)# stratify=y_train)

                partitioned_labels = [final_labels[doc] for doc in train + validation + test]
                partitioned_corpus = [final_docs[doc] for doc in train + validation + test]
                document_indexes = [document_indexes[doc] for doc in train + validation + test]
                metadata["last-training-doc"] = len(train)
                metadata["last-validation-doc"] = len(validation) + len(train)
                if self.save_original_indexes:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata,
                                   labels=partitioned_labels, document_indexes=document_indexes)
                else:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata,
                                   labels=partitioned_labels)
            else:
                train, test = train_test_split(range(len(final_docs)), test_size=0.15, random_state=1)
                train, validation = train_test_split(train, test_size=3 / 17, random_state=1)

                metadata["last-training-doc"] = len(train)
                metadata["last-validation-doc"] = len(validation) + len(train)
                partitioned_corpus = [final_docs[doc] for doc in train + validation + test]
                document_indexes = [document_indexes[doc] for doc in train + validation + test]
                if self.save_original_indexes:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                                   document_indexes=document_indexes)
                else:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                                   document_indexes=document_indexes)
        else:
            if self.save_original_indexes:
                return Dataset(final_docs, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                               document_indexes=document_indexes)
            else:

                return Dataset(final_docs, vocabulary=vocabulary, metadata=metadata, labels=final_labels)

    def filter_words(self, docs):
        if self.vocabulary is not None:
            self.preprocessing_steps.append('filter words by vocabulary')
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            vectorizer = TfidfVectorizer(df_max_freq=self.max_df, df_min_freq=self.min_df, vocabulary=self.vocabulary,
                                         token_pattern=r"(?u)\b\w{" + str(self.min_chars) + ",}\b",
                                         lowercase=self.lowercase, stop_words=self.stopwords)

        elif self.max_features is not None:
            self.preprocessing_steps.append('filter vocabulary to ' + str(self.max_features) + ' terms')
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            # we ignore df_max_freq e df_min_freq because self.max_features is not None
            vectorizer = TfidfVectorizer(lowercase=self.lowercase, max_features=self.max_features,
                                         stop_words=self.stopwords,
                                         token_pattern=r"(?u)\b[\w|\-]{" + str(self.min_chars) + r",}\b")

        else:

            #string.punctuation

            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            vectorizer = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, lowercase=self.lowercase,
                                         token_pattern=r"(?u)\b[\w|\-]{" + str(self.min_chars) + r",}\b",
                                         stop_words=self.stopwords)

        vectorizer.fit_transform(docs)
        vocabulary = vectorizer.get_feature_names_out()
        return vocabulary

    '''
    def _foo(self, docs, vocabulary, labels_path):
        final_docs, final_labels = [], []
        if labels_path is not None:
            labels = [line.strip() for line in open(labels_path, 'r').readlines()]
            for doc, label in zip(docs, labels):
                new_doc = [w for w in doc.split() if w in set(vocabulary)]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    final_labels.append(label)
            return final_docs, final_labels
        else:
            for doc in docs:
                new_doc = [w for w in doc.split() if w in set(vocabulary)]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
            return final_docs, []
    '''

    def simple_preprocessing_steps(self, doc):
        new_d = doc
        new_d = new_d.replace('\n', '')
        new_d = new_d.replace('\t', '')
        if self.lowercase:
            new_d = new_d.lower()
        if self.lemmatize:
            if self.remove_stopwords_spacy:
                new_d = ' '.join([token.lemma_ for token in self.spacy_model(new_d) if not token.is_stop])
            elif self.stopwords:
                new_d = ' '.join(
                    [token.lemma_ for token in self.spacy_model(new_d) if token.lemma_ not in set(self.stopwords)])
            else:
                new_d = ' '.join([token.lemma_ for token in self.spacy_model(new_d)])

        if self.remove_punctuation:
            new_d = new_d.translate(str.maketrans(self.punctuation, ' ' * len(self.punctuation)))
        if self.remove_numbers:
            new_d = new_d.translate(str.maketrans("0123456789", ' ' * len("0123456789")))
        new_d = " ".join(new_d.split())
        return new_d
    

class TopicModeling:
    def __init__(self, num_topics, dataset_source, learning_rate=0.001, batch_size=32, activation='softplus',
                 num_layers=3, num_neurons=100, dropout=0.2, num_epochs=100, save_model=False, model_path=None, train_model='NeuralLDA',evaluation=['accuracy','topicdiversity','invertedrbo','jaccardsimilarity','coherence']):

        """
        Initialize the TopicModeling object.

        Args:
            num_topics (int): Number of topics.

            dataset_path (str): Path to the dataset file.

            learning_rate (float): Learning rate for the model (default: 0.001).

            batch_size (int): Batch size for training (default: 32).

            activation (str): Activation function to use (default: 'softplus').

            num_layers (int): Number of layers in the model (default: 3).

            num_neurons (int): Number of neurons in each layer (default: 100).

            dropout (float): Dropout rate for regularization (default: 0.2).

            num_epochs (int): Number of epochs for training (default: 100).

            save_model (bool): Whether to save the trained model (default: False).

            model_path (str): Path to save the trained model (default: None).

            train_model (str): Model to use for training (default: 'NeuralLDA'). Other Options are 'ProdLDA', 'LDA', 'CTM'

            evaluation (str): Matrix to use for model evaluation (default: ['accuracy','topicdiversity','invertedrbo','jaccardsimilarity','coherence']).

        """

        valid_values = ['accuracy', 'topicdiversity', 'invertedrbo', 'jaccardsimilarity', 'coherence']
        assert isinstance(num_topics, int) and num_topics > 0, "num_topics should be a positive integer."
        assert isinstance(dataset_source, (str,pd.DataFrame)), "dataset_path should be a string or preprocessed dataset variable"
        assert isinstance(learning_rate, float) and learning_rate > 0, "learning_rate should be a positive float."
        assert isinstance(batch_size, int) and batch_size > 0, "batch_size should be a positive integer."
        assert activation in ['softplus', 'relu', 'sigmoid', 'swish', 'leakyrelu', 'rrelu', 'elu', 'selu', 'tanh'], "activation must be 'softplus', 'relu', 'sigmoid', 'swish', 'leakyrelu', 'rrelu', 'elu', 'selu' or 'tanh'."
        assert isinstance(num_layers, int) and num_layers > 0, "num_layers should be a positive integer."
        assert isinstance(num_neurons, int) and num_neurons > 0, "num_neurons should be a positive integer."
        assert isinstance(dropout, float) and 0 <= dropout <= 1, "dropout should be a float between 0 and 1."
        assert isinstance(num_epochs, int) and num_epochs > 0, "num_epochs should be a positive integer."
        assert isinstance(save_model, bool), "save_model should be a boolean."
        assert model_path is None or isinstance(model_path, str), "model_path should be None or a string."
        assert train_model in ['NeuralLDA', 'ProdLDA','LDA','CTM'], "train_model should be either 'NeuralLDA','ProdLDA','LDA' or 'CTM'."
        assert isinstance(evaluation, list) and all(item in valid_values for item in evaluation), "Some elements in evaluation are not valid or evaluation is not a list. Elements must be 'accuracy','topicdiversity','invertedrbo','jaccardsimilarity','coherence'."



        self.topk = num_topics
        self.dataset_path = dataset_source
        self.lr = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.save_model = save_model
        self.model_path = model_path
        self.model = globals()[train_model]
        self.t_model = str(train_model)
        self.df = None
        self.processed = None
        self.dataset = None
        self.nlda = None
        self.evaluation_results = None
        self.evaluation = evaluation

        file_size = os.path.getsize(dataset_source)
    
        # Check if the file size is less than 1 MB
        if file_size < 1024 * 1024:  # 1 MB = 1024 * 1024 bytes
            print("Recommendation: Consider using a larger file with more data (at least 1 MB).")

        

    def read_data(self):
        """
        Reads the dataset from the provided file path.
        """
        try:
            if isinstance(self.dataset_path, str):
                if self.dataset_path.endswith('.pkl'):
                    self.df = pd.read_pickle(self.dataset_path).reset_index(drop=True)
                elif self.dataset_path.endswith('.csv'):
                    self.df = pd.read_csv(self.dataset_path).reset_index(drop=True)
                else:
                    print("Error: Invalid dataset file format. Only .pkl and .csv formats are supported.")
                    return False
            else:
                self.df = self.dataset_path
        except FileNotFoundError:
            print("Error: Dataset file not found.")
            return False
        except Exception as e:
            print("Error: Reading dataset failed:", e)
            return False
        return True

    def preprocess_data(self):
        """
        Preprocesses the data by removing 'RT', dropping NaN values, and saving the processed text to a file.
        """
        
        if self.df is None:
            print("Error: No data loaded. Call read_data() first.")
            return False

        try:
            self.processed = self.df['text'].str.replace('RT', '').dropna().reset_index()['text']
            self.processed = self.processed[self.processed.str.len() > 3]
            self.processed = self.processed.astype('str')
            self.processed.to_csv('corpus.txt', header=False, index=False, sep='\t')
        except Exception as e:
            print("Error: Preprocessing data failed:", e)
            return False
        return True

    @staticmethod
    def text_process(keyword):
        """
        Takes in a keyword and performs text preprocessing by removing punctuation and stopwords.
        """
        # Remove punctuation
        nopunc = ''.join(char for char in keyword if char not in string.punctuation)

        # Remove stopwords
        cleaned_keyword = ' '.join(word for word in nopunc.split() if word.lower() not in stopwords.words('english'))

        return cleaned_keyword

    def generate_labels(self):
        """
        Generates labels by applying text processing to keywords and saving them to a file.
        """
        if self.df is None:
            print("Error: No data loaded. Call read_data() first.")
            return False

        try:
            with open('labels.txt', 'w') as file:
                unique_keywords = set()
                for i in self.df['text_tokens']:
                    for j in i:
                        keyword = j.strip(" []'")
                        cleaned_keyword = self.text_process(keyword)
                        if cleaned_keyword:
                            unique_keywords.add(cleaned_keyword)
                            file.write(cleaned_keyword + '\n')
        except Exception as e:
            print("Error: Generating labels failed:", e)
            return False
        return True

    def preprocess_dataset(self):
        """
        Preprocesses the dataset by applying additional preprocessing steps and creating an OCTIS dataset.
        """
        try:
            preprocessor = Preprocessing(vocabulary=None, max_features=None,
                                         remove_punctuation=True, punctuation=string.punctuation,
                                         lemmatize=True, stopword_list='english',
                                         min_chars=3, min_words_docs=0)

            self.dataset = preprocessor.preprocess_dataset(documents_path='corpus.txt', labels_path='labels.txt')
        except FileNotFoundError:
            print("Error: Preprocessed dataset file not found.")
            return False
        return True

    def train_model(self):
        """
        Trains the NeuralLDA model using the preprocessed dataset.
        """
        if self.dataset is None:
            print("Error: No preprocessed dataset. Call preprocess_dataset() first.")
            return False

        try:
            if self.model == LDA:
                model = self.model(num_topics=self.topk,  chunksize=self.batch_size,
                                decay=self.dropout, iterations=self.num_epochs,gamma_threshold=self.lr,random_state=42)
                
            else:
                model = self.model(num_topics=self.topk, lr=self.lr, batch_size=self.batch_size, activation=self.activation,
                                dropout=self.dropout, num_epochs=self.num_epochs,
                                num_layers=self.num_layers, num_neurons=self.num_neurons)
            

            self.nlda = model.train_model(self.dataset)

            
            
        except Exception as e:
            print("Error: Training model failed:", e)
            return False
        return True


    def save_trained_model(self):
        """
        Saves the trained model to a file if the 'save_model' attribute is set to True.
        """
        if self.save_model:
            
            if self.nlda is None:
                print("Error: No trained model. Call train_model() first.")
                return False

            if self.model_path is None:
                pkl_filename = self.model+'_trained_model.pkl'
            else:
                pkl_filename = self.model_path+str(self.t_model)+'_trained_model.pkl'

            try:
                
                with open(pkl_filename, 'wb') as file:
                    pickle.dump(self.nlda, file)
            except Exception as e:
                print("Error: Saving model failed:", e)
                return False
        return True

    def evaluate_model(self):
        """
        Evaluates the trained model using various evaluation metrics and displays the scores.
        """
        if self.nlda is None:
            print("Error: No trained model. Call train_model() first.")
            return False

        def gradient_color(value):
            start_color = (255, 0, 0)  # Red
            end_color = (0, 255, 0)    # Green
            r = int(start_color[0] + (end_color[0] - start_color[0]) * value)
            g = int(start_color[1] + (end_color[1] - start_color[1]) * value)
            b = int(start_color[2] + (end_color[2] - start_color[2]) * value)
            return f'\033[38;2;{r};{g};{b}m'

        def print_colored_progress_bar(value):
            bar_length = 40
            progress = int(value * bar_length)
            bar = '█' * progress + '-' * (bar_length - progress)
            percentage = int(value * 100)
            color_code = gradient_color(value)
            reset_color = '\033[0m'
            return f'{color_code}[{bar}] {percentage}%{reset_color}'  # Return the formatted string

        try:
            try:
                topic_diversity_score = TopicDiversity(topk=self.topk).score(self.nlda)
            except Exception as e:
                print("Error: topic diversity failed:", e)
            try:
                inverted_rbo_score = InvertedRBO(topk=self.topk).score(self.nlda)
            except Exception as e:
                print("Error: inverted rbo failed:", e)
            
            try:
                accuracy_score = AccuracyScore(self.dataset).score(self.nlda)
            except Exception as e:
                print("Error: accuracy score failed:", e)
            try:
                pairwise_jaccard_similarity_score = PairwiseJaccardSimilarity(topk=self.topk).score(self.nlda)
            except Exception as e:
                print("Error: pairwise jaccard similarity score failed:", e)
            try:
                coherence_score = Coherence(texts=self.nlda['topics'], topk=self.topk, measure='c_v').score(self.nlda)
            except Exception as e:
                print("Error: coherence score failed:", e)

            
            # Your evaluation code goes here, and the results are stored in self.evaluation_results

            # Assuming you have already set the evaluation results as shown in your code
            
            self.evaluation_results = ' Model Evaluation '
            if 'accuracy' in self.evaluation:
                self.evaluation_results += '\n\nAccuracy Score: {}\t{:.4f}'.format(print_colored_progress_bar(accuracy_score), accuracy_score)
            if 'topicdiversity' in self.evaluation:
                self.evaluation_results += '\nTopic Diversity Score: {}\t{:.4f}'.format(print_colored_progress_bar(topic_diversity_score), topic_diversity_score)
            if 'invertedrbo' in self.evaluation:
                self.evaluation_results += '\nInverted RBO Score: {}\t{:.4f}'.format(print_colored_progress_bar(inverted_rbo_score), inverted_rbo_score)
            if 'jaccardsimilarity' in self.evaluation:
                self.evaluation_results += '\nPairwise Jaccard Similarity Score: {}\t{:.4f}'.format(print_colored_progress_bar(1 - pairwise_jaccard_similarity_score), pairwise_jaccard_similarity_score)
            if 'coherence' in self.evaluation:
                self.evaluation_results += '\nCoherence Score: {}\t{:.4f}'.format(print_colored_progress_bar(coherence_score), coherence_score)
            
            print('')
            print('')
            print(str(self.evaluation_results).strip())  # Remove extra whitespace at the end

        except Exception as e:
            print("Error: Evaluating model failed:", e)
            return False
        return True

    def run(self):
        """
        Executes the pipeline by calling each step in order.
        """
        if not self.read_data():
            return False
        if not self.preprocess_data():
            return False
        if not self.generate_labels():
            return False
        if not self.preprocess_dataset():
            return False
        if not self.train_model():
            return False
        if not self.save_trained_model():
            return False
        if not self.evaluate_model():
            return False
        
        return self.nlda
        
        


