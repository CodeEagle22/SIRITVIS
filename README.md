

# SIRITVIS

Social Interaction Research Insights Topic Visualization

## Summary   

The integration of the SIRITVIS Python package offers a robust and scientifically grounded approach for getting insights of data from platforms like Twitter and Reddit. It provides a comprehensive set of features including data streaming, preprocessing, model training, topic evaluation metrics, topic distribution and graph visualisation, and geo-visualization tools. By leveraging these capabilities, organizations can gain valuable insights into the spatial distribution of topics and make data-driven decisions.

The package combines established methodologies from data science, machine learning, and geospatial analysis, ensuring the reliability and accuracy of the results. Rigorous preprocessing techniques and model training enhance the validity of the extracted topics, while scientifically validated evaluation metrics assess their quality and relevance. The geo-visualization tools enable clear and intuitive understanding of the spatial distribution of topics.

Overall, the adoption of the SIRITVIS package empowers organizations across domains such as marketing, politics, and disaster management to deepen their understanding of customers and stakeholders. By leveraging spatial topic distribution insights, they can enhance engagement and facilitate informed decision-making processes, leading to improved outcomes.

## How to cite
Narwade, S., Kant, G., and Säfken, B. (2020), SIRITVIS: Social Interaction Research Insights Topic Visualization. SoftwareX, 5 (54), 2507, https://doi.org/10.21105/joss.02507.


## Features

- Data Streaming 💾
- Data Cleaning 🧹
- Topic Model Training and Evaluation :dart:
- Topic Visual Insights 🔍
- Twitter Topic Geo Visualisation 🌏

## 🛠 Installation

Attention: SIRITVIS is designed to run on Python 3.10, it has been fully tested under these conditions. We recommend installing a new (conda) environment using Python 3.10.10 for optimal compatibility and performance.

The package can be installed via pip:

```bash
cd ../dist
pip install SIRITVIS-1.0.tar.gz
```

## 👩‍💻 Usage ([documentation])

### Import Libraries

```python
from SIRITVIS import reddit_streamer, cleaner, topic_model, topic_visualise, tweet_mapper
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

auth_path = '../file/path/to/load/the/credentials.txt'
save_path = '../folder/path/to/store/the/data/'
languages = ['en'] # could select multiple languages
locations = [51.416016,5.528511,90.966797,34.669359] # box coordinates.
keywords = ['Specific','Keywords'] # default is None
raw_data = twitter_streamer.TwitterStreamer(auth_path=auth_path,save_path=save_path,languages=languages,locations=locations,keywords=keywords)
```

### Cleaning Reddit, Twitter or Any External Text Data

```python
# raw_data variable could also used as load_path attribute value

clean_data = cleaner.Cleaner(data_source='../folder/path/or/csv/file/path/to/load/data/',data_save_name='twitter',data='twitter')

cleaned_file = clean_data.saving('../folder/path/to/store/the/cleaned/data/')
```

### Train your a topic model on a corpus of short texts

```python
# cleaned_file variable could also used as dataset_path attribute value

model = topic_model.TopicModeling(num_topics=10, dataset_source='../csv/file/path/to/load/data.csv',
learning_rate=0.001, batch_size=32, activation='softplus', num_layers=3, num_neurons=100,
dropout=0.2, num_epochs=100, save_model=False, model_path=None, train_model='NeuralLDA')

saved_model = model.run()
```

### Topic Insights Visualisation 

To investigate internal structure of topics and their relations to words and indicidual documents we recommend using [pyLDAvis](https://github.com/bmabey/pyLDAvis).
```python
# cleaned_file variable could also used as file_path attribute value

vis_model = topic_visualise.PyLDAvis(data_source='../csv/file/path/to/load/data.csv',num_topics=5,text_column='text')
vis_model.visualize()
```

To investigate internal structure of topics and their relations to words and indicidual documents we recommend using [topicwizard]
(https://github.com/x-tabdeveloping/topic-wizard).
```python
# cleaned_file variable could also used as csv_file attribute value
# please wait for a while for the word cloud to appear.

vis_model = topic_visualise.TopicWizardvis(data_source='../csv/file/path/to/load/data.csv',num_topics=5,text_column='text')
vis_model.visualize()
```


### Twitter Topic Geo Visualisation 

```python
# cleaned_file variable could also used as csv_file_path attribute value
# saved_model variable could also used as model_file attribute value

csv_file_path = '../file/path/of/data.csv'
model_path = '../file/path/of/model.pkl'
tweet_mapper.TweetMapper(csv_file_path,model_path)
```
## Community guidelines

We encourage and welcome contributions to the SIRITVIS package. If you have any questions, want to report bugs, or have ideas for new features, please file an issue. 

Additionally, we appreciate pull requests via GitHub. There are several areas where potential contributions can make a significant impact, such as enhancing the quality of topics in topic models when dealing with noisy data from Reddit and Twitter or any external data sources, and improving the tweet_mapper function to make it more interactive and independent from the notebook.

## Authors

Sagar Narwade
Gillian Kant
Benjamin Säfken

## License
SIRITVIS is published under the GNU GPLv3 license.



## 🎓 References


