

# SIRITVIS

Social Interaction Research Insights Topic Visualization

## Summary   

The package SIRITVIS provides a broad range of methods to generate, clean, analyze and visualize the contents of Social Media Data (Reddit and Twitter). SIRITVIS enables the user to work with geo-spatial Twitter data and to generate topic distributions from NeuralLDA and ProdLDA Topic Models for geo-coded Tweets. As such, SIRITVIS is an innovative 
tool to work with geo-coded text on a high geo-spatial resolution to analyse the public discourse on various topics in 
space and time. The package can be used for a broad range of applications for scientific research to gain insights into 
topics discussed on Twitter. 

In general, Topic Models are generative probabilistic models, that provide an insight into hidden information 
in large text corpora by estimating the underlying topics of the texts in an unsupervised manner.

Firstly, the package allows the user to collect Tweets using a Twitter developer account for any area in the world.
Subsequently, the inherently noisy Twitter data can be cleaned, transformed and exported. 
In particular, TTLocVis enables the user to apply LDA Topic Models on extremely sparse Twitter data by preparing 
the Tweets for LDA analysis by the pooling Tweets by hashtags.

TTLocVis provides options for automatized Topic Model parameter optimization. Furthermore, a distribution over 
topics is generated for each document. The distribution of topics over documents can be visualized with various 
plotting methods. The average prevalence of topics in the documents at each day can 
be plotted as a time series, in order to visualize, how topics develop over time.
 
Above this, the spatial distribution of Tweets can be plotted on a world map, which automatically chooses an appropriate
part of the world, in order to visualise the chosen sample of Tweets. As part of the mapping process, each Tweet is 
classified by its most prevalent topic and colour coded.

## Features

- Data Streaming 💾
- Data Cleaning 🧹
- Topic Model Training and Evaluation :dart:
- Topic Visual Insights 🔍
- Twitter Topic Geo Visualisation 🌏

## 🛠 Installation

Install from PyPI:

```bash
cd ../dist
pip install SIRITVIS-1.0.tar.gz
```

## 👩‍💻 Usage ([documentation](https://centre-for-humanities-computing.github.io/tweetopic/))

Import Libraries

```python
from SIRITVIS import reddit_streamer, cleaner, topic_model, topic_visualise, tweet_mapper
```

Streaming Reddit Data

```python
client_id = "XXXXXXXXXX"
client_secret = "XXXXXXXXX"
user_agent = "XXXXXXXXXX"
keywords = ['Specific','Keywords']
save_path = '../folder/path/to/store/the/data/'
raw_data = reddit_streamer.RedditStreamer(client_id,client_secret,user_agent,save_path,keywords).run()
```

Cleaning Reddit, Twitter or Any External Text Data

```python
# raw_data variable could also used as load_path attribute value

clean_data = cleaner.Cleaner(load_path='../folder/path/or/csv/file/path/to/load/data/',data_save_name='twitter',data='twitter')

cleaned_file = clean_data.saving('../folder/path/to/store/the/cleaned/data/')
```

Train your a topic model on a corpus of short texts

```python
# cleaned_file variable could also used as dataset_path attribute value

model = topic_model.TopicModeling(num_topics=10, dataset_path='../csv/file/path/to/load/data.csv',
learning_rate=0.001, batch_size=32, activation='softplus', num_layers=3, num_neurons=100,
dropout=0.2, num_epochs=100, save_model=False, model_path=None, train_model='NeuralLDA')

saved_model = model.run()
```

Topic Insights Visualisation 

To investigate internal structure of topics and their relations to words and indicidual documents we recommend using [pyLDAvis](https://github.com/bmabey/pyLDAvis).

```python
# cleaned_file variable could also used as file_path attribute value

vis_model = topic_visualise.PyLDAvis(file_path='../csv/file/path/to/load/data.csv',text_column='text')
vis_model.visualize()
```
To investigate internal structure of topics and their relations to words and indicidual documents we recommend using [pyLDAvis](https://github.com/bmabey/pyLDAvis).

```python
# cleaned_file variable could also used as csv_file attribute value

vis_model = topic_visualise.TopicWizardvis(csv_file='../csv/file/path/to/load/data.csv',num_topics=5,text_column='text)
vis_model.visualize()
```

To investigate internal structure of topics and their relations to words and indicidual documents we recommend using [topicwizard](https://github.com/x-tabdeveloping/topic-wizard).

Twitter Topic Geo Visualisation 

```python
csv_file_path = '../file/path/of/data.csv'
model_file = '../file/path/of/model.pkl'
tweet_mapper.TweetMapper(csv_file_path,model_file)
```




## 🎓 References

- Yin, J., & Wang, J. (2014). A Dirichlet Multinomial Mixture Model-Based Approach for Short Text Clustering. _In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 233–242). Association for Computing Machinery._
