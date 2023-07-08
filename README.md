

# SIRITVIS


## Features

- Data Extraction 💾:
- Data Preprocessing 🧹:
- Topic Model Training and Evaluation :dart:
- Topic Visual Insights :eyes:

## 🛠 Installation

Install from PyPI:

```bash
cd dist
pip install SIRITVIS-1.0.tar.gz
```

## 👩‍💻 Usage ([documentation](https://centre-for-humanities-computing.github.io/tweetopic/))

Import Libraries

```python
from SIRITVIS import reddit_streamer, reddit_cleaner, topic_model, topic_visualise
```

Streaming Reddit Data
```python
client_id = "XXXXXXXXXX"
client_secret = "XXXXXXXXX"
user_agent = "XXXXXXXXXX"
keywords = ['Specific','Keywords']
save_path = 'Path/Directory/Store/Data/'
RedditStreamer(client_id,client_secret,user_agent,save_path,keywords)
```

Train your a topic model on a corpus of short texts:

```python
from tweetopic import DMM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# Creating a vectorizer for extracting document-term matrix from the
# text corpus.
vectorizer = CountVectorizer(min_df=15, max_df=0.1)

# Creating a Dirichlet Multinomial Mixture Model with 30 components
dmm = DMM(n_components=30, n_iterations=100, alpha=0.1, beta=0.1)

# Creating topic pipeline
pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("dmm", dmm),
])
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
