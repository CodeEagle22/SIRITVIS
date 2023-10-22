

# SIRITVIS

Social Interaction Research Insights Topic Visualisation

## Summary   

The SIRITVIS Python package offers a robust and scientifically grounded approach for gaining insights from data on platforms like Twitter, Instagram and Reddit. By leveraging advanced Topic Models, including Latent Dirichlet Allocation (LDA), Neural Latent Dirichlet Allocation (NeuralLDA), Prod Latent Dirichlet Allocation (ProdLDA), and CTM Topic Models, SIRITVIS enables users to identify hidden patterns in vast text corpora in an unsupervised manner. The package provides a comprehensive set of features, including data streaming, preprocessing, model training, topic evaluation metrics, topic distribution, graph visualisation, and geo-visualisation tools. These capabilities allow organisations to extract valuable insights and make data-driven decisions.

The integration of established methodologies from data science, machine learning, and geospatial analysis ensures the reliability and accuracy of the results. Rigorous preprocessing techniques and model training enhance the validity of the extracted topics, while scientifically validated evaluation metrics assess their quality and relevance. The graph visualisation and geo-visualisation tools facilitate a clear and intuitive understanding of the spatial distribution of topics.

One of the standout feature of SIRITVIS is its ability to map the spatial distribution of Tweets and Instagram post on a world map, associating each location with its top trending topics and their frequency. The package classifies and color-codes locations based on their sentiments, providing a comprehensive count of positive, negative, and neutral tweets. Furthermore, users can explore specific keywords through a convenient dropdown interface and visualise their occurrences on the world map.

The innovative capabilities of this tool hold great potential in various domains, such as marketing, politics, and disaster management, empowering data-driven decision-making through spatial topic distribution insights. Organisations can leverage SIRITVIS to gain a deeper understanding of their customers and stakeholders, foster engagement, and facilitate informed decision-making processes based on comprehensive social media data analysis.

## How to cite
Narwade, S., Kant, G., Säfken, B., and Leiding, B. (2023), SIRITVIS: Social Interaction Research Insights Topic Visualisation.


## Features

- Data Streaming 💾
- Data Cleaning 🧹
- Topic Model Training and Evaluation :dart:
- Topic Visual Insights 🔍
- Trending Topic Geo Visualisation 🌏

## 🛠 Installation

Attention: SIRITVIS is designed to run on Python 3.10, it has been fully tested under these conditions. We recommend installing a new (conda) environment using Python 3.10.10 for optimal compatibility and performance.

The package can be installed via pip:

```bash
pip install SIRITVIS
```

## 👩‍💻 Usage ([documentation])

### Import Libraries

```python
from SIRITVIS import twitter_streamer, insta_streamer, reddit_streamer, cleaner, topic_model, topic_visualise, topic_mapper
```

### Streaming Reddit Data

```python
# Run the streaming process to retrieve raw data based on the specified keywords

client_id = "XXXXXXXXXX"
client_secret = "XXXXXXXXX"
user_agent = "XXXXXXXXXX"
keywords = ['Specific','Keywords'] # default is None
save_path = '../folder/path/to/store/the/data/'
raw_data = reddit_streamer.RedditStreamer(client_id,client_secret,user_agent,save_path,keywords).run()
```

### Streaming Twitter Data

```python
# Run the streaming process to retrieve raw data based on the specified keywords and for specific location

# Twitter API credentials
consumer_key = 'XXXXXXXXX'
consumer_secret = 'XXXXXXXXX'
access_token = 'XXXXXXXXX'
access_secret = 'XXXXXXXXX'

# Location and language settings
locations = [51.416016,5.528511,90.966797,34.669359] # box coordinates.
languages = ['en', 'es'] 

# Keywords to track
keywords = ['Specific','Keywords'] # default is None

# Save path for collected data
save_path = '../folder/path/to/store/the/data/'

# Initialise and start Twitter streamer
raw_data = twitter_streamer.TwitterStreamer(
    consumer_key,
    consumer_secret,
    access_token,
    access_secret,
    languages,  
    locations=locations,
    keywords=keywords,
    save_path=save_path  # Make sure to provide the save_path argument correctly
)
```

### Streaming Instagram Data

```python
# Run the streaming process to retrieve raw data based on the specified keywords
# Sign_up [apify](https://apify.com/apify/instagram-hashtag-scraper) to get free api_token

api_token = 'apify_api_XXXXXXXXX'
save_path = '../folder/path/to/store/the/data/'
instagram_username = 'XXXXXXXXX'
instagram_password = 'XXXXXXXXX'
hashtags = ['Specific','Keywords'] # default is ['instagram']
limit =  20 # number of post captions to extract. default is 100
raw_data  = insta_streamer.InstagramStreamer(api_token,save_path,instagram_username,instagram_password,hashtags,limit).run()
```

### Clean Streamed Data or Any External Text Data

```python
# raw_data variable might also be used as load_path attribute value
cleaner_obj = cleaner.Cleaner(data_source='../folder/path/or/csv/file/path/to/load/data/',data_source_type='twitter or default:None')
# cleaner_obj.clean_data     # get cleaned dataset without saving it
cleaned_file = cleaner_obj.saving('../folder/path/to/store/the/cleaned/data/',data_save_name='dataset_file_name')
```

### Train your a topic model on corpus of short texts

```python
# cleaned_file variable might also be used as dataset_source attribute value

model = topic_model.TopicModeling(num_topics=10, dataset_source='../csv/file/path/to/load/data.csv',
learning_rate=0.001, batch_size=32, activation='softplus', num_layers=3, num_neurons=100,
dropout=0.2, num_epochs=100, save_model=False, model_path=None, train_model='NeuralLDA',evaluation=['topicdiversity','invertedrbo','jaccardsimilarity'])

saved_model = model.run()
```

### Topic Insights Visualisation 

To investigate internal structure of topics and their relations to words and indicidual documents we recommend using [pyLDAvis](https://github.com/bmabey/pyLDAvis).
```python
# cleaned_file variable could also used as data_source attribute value

vis_model = topic_visualise.PyLDAvis(data_source='../csv/file/path/to/load/data.csv',num_topics=5,text_column='text')
vis_model.visualize()
```

A graphical display of text data in which the importance of each word reflects its frequency or significance within the text.
```python
# The cleaned_file variable might also be used as data_source attribute value
# please wait for a while for the word cloud to appear.

vis_model = topic_visualise.Wordcloud(data_source='../csv/file/path/to/load/data.csv',text_column='text')
vis_model.visualize()
```


### Trending Topic Geo Visualisation 

Topic Mapper excels at mapping the spatial distribution of tweets and Instagram posts globally. It accomplishes this by associating each location with its top trending topics and their frequencies, all using pre-trained topic models. Furthermore, it categorizes and color-codes these locations based on sentiment, providing users with a quick overview of sentiment distribution, including counts for positive, negative, and neutral tweets.

Users can effortlessly explore specific keywords through a dropdown interface, allowing them to see how frequently these keywords appear on the world map. This feature simplifies the process of grasping and navigating research findings.

```python
# The cleaned_file variable might also be used as data_source attribute value
# The saved_model variable might also be used as the model_source attribute value, for example, model_source = saved_model

data_source = '../file/path/of/data.csv'
model_source = '../file/path/of/model.pkl' 
topic_mapper.TopicMapper(data_source, model_source)
```

## Community guidelines

We encourage and welcome contributions to the SIRITVIS package. If you have any questions, want to report bugs, or have ideas for new features, please file an issue. 

Additionally, we appreciate pull requests via GitHub. There are several areas where potential contributions can make a significant impact, such as enhancing the quality of topics in topic models when dealing with noisy data from Reddit, Instagram and Twitter or any external data sources, and improving the topic_mapper function to make it more interactive and independent from the notebook.

## Authors

Sagar Narwade, 
Gillian Kant,
Benjamin Säfken,
Benjamin Leiding,

## 🎓 References
In our project, we utilised the "OCTIS" [^1^] tool, a fantastic library by Terragni et al., which provided essential functionalities. Additionally, we incorporated the "pyLDAvis" [^2^] by Ben Mabey Python library for interactive topic model visualisation, enriching our application with powerful data insights. The seamless integration of these resources significantly contributed to the project's success, offering an enhanced user experience and valuable research capabilities.

[^1^]: [OCTIS](https://github.com/MIND-Lab/OCTIS).
[^2^]: [pyLDAvis](https://github.com/bmabey/pyLDAvis)

