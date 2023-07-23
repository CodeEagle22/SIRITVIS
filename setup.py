from setuptools import setup, find_packages
import warnings



# Set the warning filter to "ignore"
warnings.filterwarnings("ignore")
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    print("README.md file not found.")
    long_description = ""
except Exception as e:
    print("An error occurred while reading README.md:", str(e))
    long_description = ""

setup(
    name='SIRITVIS',
    version='1.0',
    author='Sagar Narwade, Gillian Kant, Benjamin Saefken',
    description="SIRITVIS: Social Media Interaction & Reaction Insights Topic Visualisation",
    maintainer="Sagar Narwade",
    maintainer_email="sagarnarwade147@gmail.com",
    python_requires='==3.10.*',
    packages=find_packages(),
    install_requires=[
        'praw==7.7.1',
        'numpy',
        'pandas==1.5.3',
        'spacy',
        'gensim',
        'pyLDAvis==3.4.0',
        'matplotlib',
        'tweepy==3.9.0',
        'urllib3',
        'wordcloud',
        'langdetect',
        'octis',
        'topic-wizard==0.3.1',
        'tweepy==3.9.0',
        'scikit-learn==1.2.2',
        'urllib3',
        'langdetect',
        'folium==0.14.0',
        'pickle4==0.0.1',
        'plotly==5.15.0',
        'nltk==3.6.2',
        'notebook==6.4.8',
        'ipywidgets',
        'tensorflow',
        'vaderSentiment',
        'ipyleaflet',
        'geopy',
        'Flask==2.2.5',
        'Flask-Caching==2.0.1',
        'wordcloud==1.8.2.2',
        'ipython==7.34.0',
        'tweepy==3.9.0',
        'textblob==0.15.3'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.10',
    ],
    
)

