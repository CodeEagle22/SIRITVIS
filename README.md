

# SIRITVIS

Social Interaction Research Insights Topic Visualisation


<img src="images/image.png" alt="Logo" height="180" width="180">


## 📋 Summary   

The SIRITVIS Python package helps you understand data from social media platforms like Instagram, Reddit or any other text data sources. It uses advanced techniques to find hidden patterns in large amounts of text data. SIRITVIS includes tools for gathering data, cleaning it, analyzing it, and visualizing the results. You can see where certain topics are being talked about on a map and how often they are mentioned.

SIRITVIS uses well-known methods from data science, machine learning, and mapping to ensure accurate results. It cleans the data thoroughly and uses reliable models to find meaningful topics. You can evaluate the quality of these topics using built-in tools. The package also includes visual tools to help you easily see the distribution of topics on a map.

A key feature of SIRITVIS is its ability to show where on a world map people are talking about different topics. It can categorize these places by the sentiment of the posts, such as positive, negative, or neutral. You can also search for specific keywords and see where they appear on the map.

SIRITVIS is helpful in various areas, like marketing, politics, and disaster response, by providing tools to analyze the spread of topics. It helps users understand their audience better and make informed decisions based on the analysis of social media data.

## 📝 How to cite

Narwade, S., Kant, G., Säfken, B., and Leiding, B. (2023), SIRITVIS: Social Interaction Research Insights Topic Visualisation. Journal of Open Source Software, https://joss.theoj.org/papers/b51be70e9634e45d8035ee20b6147d76.

## Markdown:
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06243/status.svg)](https://doi.org/10.21105/joss.06243)

HTML:
<a style="border-width:0" href="https://doi.org/10.21105/joss.06243">
  <img src="https://joss.theoj.org/papers/10.21105/joss.06243/status.svg" alt="DOI badge" >
</a>





## Advisory

- Ensure Python version '>=3.10, <3.11'.
- Utilize IDEs like Visual Studio or platforms like Google Colab for enhanced plot visualization.
- Refer to the provided [sample dataset](https://github.com/CodeEagle22/SIRITVIS/tree/main/sample_dataset) for better comprehension.

## 💡 Features

- Data Streaming 💾
- Data Cleaning 🧹
- Topic Model Training and Evaluation :dart:
- Topic Visual Insights 🔍
- Trending Topic Geo Visualisation 🌏

## 🛠 Installation

Attention: SIRITVIS is specifically tailored for operation on Python 3.10, and its visualization capabilities are optimized for Python notebooks. Extensive testing has been conducted under these specifications. For the best compatibility and performance, we advise setting up a fresh (conda) environment utilizing Python 3.10.10.

The package can be installed via pip:

```bash
pip install SIRITVIS
```

## 👩‍💻 Usage ([documentation])

### Import Libraries

```python
from SIRITVIS import insta_streamer, reddit_streamer, cleaner, topic_model, topic_visualise, topic_mapper
```

### Streaming Reddit Data
- For authentication with the Reddit Streaming API, follow the steps outlined in this [tutorial](https://towardsdatascience.com/how-to-use-the-reddit-api-in-python-5e05ddfd1e5c).

```python
# Run the streaming process to retrieve raw data based on the specified keywords

client_id = "XXXXXXXXXX"
client_secret = "XXXXXXXXX"
user_agent = "XXXXXXXXXX"
keywords = ['Specific','Keywords'] # default is None # Use multiple keywords for a more varied dataset during streaming data collection.
save_path = '../folder/path/to/store/the/data/'
raw_data = reddit_streamer.RedditStreamer(client_id,client_secret,user_agent,save_path,keywords).run()
```


### Streaming Instagram Data
- For authentication with the Instagram Streaming API, sign up the page [apify](https://apify.com/apify/instagram-hashtag-scraper)

```python
# Run the streaming process to retrieve raw data based on the specified keywords

api_token = 'apify_api_XXXXXXXXX'
save_path = '../folder/path/to/store/the/data/'
instagram_username = 'XXXXXXXXX'
instagram_password = 'XXXXXXXXX'
hashtags = ['Specific','Keywords'] # default is ['instagram'] # Use multiple keywords for a more varied dataset during streaming data collection.
limit =  20 # number of post captions to extract. default is 100
raw_data  = insta_streamer.InstagramStreamer(api_token,save_path,instagram_username,instagram_password,hashtags,limit).run()
```

### Clean Streamed Data or Any External Text Data

```python
# raw_data variable might also be used as load_path attribute value
cleaner_obj = cleaner.Cleaner(data_source='../folder/path/or/csv/file/path/to/load/data/')
# cleaner_obj.clean_data     # get cleaned dataset without saving it
cleaned_file = cleaner_obj.saving('../folder/path/to/store/the/cleaned/data/',data_save_name='dataset_file_name')
```

### Train your a topic model on corpus of short texts
- Recommendation: Consider using a larger cleaned file with more data (at least 500 KB)
  
```python
# cleaned_file variable might also be used as dataset_source attribute value

model = topic_model.TopicModeling(num_topics=10, dataset_source='../csv/file/path/to/load/data.csv',
learning_rate=0.001, batch_size=32, activation='softplus', num_layers=3, num_neurons=100,
dropout=0.2, num_epochs=100, save_model=False, model_path=None, train_model='NeuralLDA',evaluation=['topicdiversity','invertedrbo','jaccardsimilarity'])

saved_model = model.run()
```

### Topic Insights Visualisation 
- To investigate internal structure of topics and their relations to words and indicidual documents we recommend using [pyLDAvis](https://github.com/bmabey/pyLDAvis).
- Recommendation: Consider using a larger cleaned file with more data (at least 500 KB)

```python
# cleaned_file variable could also used as data_source attribute value

vis_model = topic_visualise.PyLDAvis(data_source='../csv/file/path/to/load/data.csv',num_topics=5,text_column='text')
vis_model.visualize()
```

A graphical display of text data in which the importance of each word reflects its frequency or significance within the text.
- Recommendation: Consider using a larger cleaned file with more data (at least 500 KB)

```python
# The cleaned_file variable might also be used as data_source attribute value
# please wait for a while for the word cloud to appear.

vis_model = topic_visualise.Wordcloud(data_source='../csv/file/path/to/load/data.csv',text_column='text',save_image=False)
vis_model.visualize()
```


### Trending Topic Geo Visualisation 

Topic Mapper excels at mapping the spatial distribution of Instagram posts and other text data globally. It accomplishes this by associating each location with its top trending topics and their frequencies, all using pre-trained topic models. Furthermore, it categorizes and color-codes these locations based on sentiment, providing users with a quick overview of sentiment distribution, including counts for positive, negative, and neutral posts.

Users can effortlessly explore specific keywords through a dropdown interface, allowing them to see how frequently these keywords appear on the world map. This feature simplifies the process of grasping and navigating research findings.

- Notice: Reddit data cannot be visualized on the topic_mapper due to the absence of coordinate values.
  
```python
# The cleaned_file variable might also be used as data_source attribute value
# The saved_model variable might also be used as the model_source attribute value, for example, model_source = saved_model

data_source = '../file/path/of/data.csv'
model_source = '../file/path/of/model.pkl' 
topic_mapper.TopicMapper(data_source, model_source)
```

## 📣 Community guidelines

We encourage and welcome contributions to the SIRITVIS package. If you have any questions, want to report bugs, or have ideas for new features, please file an issue. 

Additionally, we appreciate pull requests via GitHub. There are several areas where potential contributions can make a significant impact, such as enhancing the quality of topics in topic models when dealing with noisy data from Reddit, Instagram or any external data sources, and improving the topic_mapper function to make it more interactive and independent from the notebook.

## 🖊️ Authors

- Sagar Narwade
- Gillian Kant
- Benjamin Säfken
- Benjamin Leiding

## 🎓 References
In our project, we utilised the "OCTIS" [^1^] tool, a fantastic library by Terragni et al., which provided essential functionalities. Additionally, we incorporated the "pyLDAvis" [^2^] by Ben Mabey Python library for interactive topic model visualisation, enriching our application with powerful data insights. The seamless integration of these resources significantly contributed to the project's success, offering an enhanced user experience and valuable research capabilities.

[^1^]: [OCTIS](https://github.com/MIND-Lab/OCTIS).
[^2^]: [pyLDAvis](https://github.com/bmabey/pyLDAvis)

## 📜 License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See the [LICENSE](./LICENSE) file for details.




