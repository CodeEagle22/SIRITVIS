

# SIRITVIS


## Features

- Data Streaming 💾
- Data Cleaning 🧹
- Topic Model Training and Evaluation :dart:
- Topic Visual Insights 🔍

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
streamer.RedditStreamer(client_id,client_secret,user_agent,save_path,keywords)
```

Cleaning Reddit, Twitter or Any External Text Data

```python
clean_data = cleaner.Cleaner('../folder/path/or/csv/file/path/to/load/data/',data_save_name='twitter',data='twitter')

clean_data.saving('../folder/path/to/store/the/cleaned/data/')
```

Train your a topic model on a corpus of short texts

```python
model = topic_model.TopicModeling(num_topics=10, dataset_path='/content/drive/MyDrive/Thesis Pipeline/raw_data_json/reddit_part_1.csv', learning_rate=0.001, batch_size=32, activation='softplus', num_layers=3, num_neurons=100, dropout=0.2, num_epochs=100, save_model=False, model_path=None, train_model='NeuralLDA')

model.run()
```

You may fit the model with a stream of short texts:

```python
pipeline.fit(texts)
```

To investigate internal structure of topics and their relations to words and indicidual documents we recommend using [topicwizard](https://github.com/x-tabdeveloping/topic-wizard).

Install it from PyPI:

```bash
pip install topic-wizard
```

Then visualize your topic model:

```python
import topicwizard

topicwizard.visualize(pipeline=pipeline, corpus=texts)
```

![topicwizard visualization](docs/_static/topicwizard.png)

## 🎓 References

- Yin, J., & Wang, J. (2014). A Dirichlet Multinomial Mixture Model-Based Approach for Short Text Clustering. _In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 233–242). Association for Computing Machinery._
