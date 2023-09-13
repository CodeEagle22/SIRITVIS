---
title: 'SIRITVIS: Social Interaction Research Insights Topic Visualization'
tags:
  - Python
  - Text analysis tool
  - Twitter, Reddit, Instagram
  - Topic Modelling
  - Geospatial mapping
  - Natural Language Processing
  - Machine Learning
 
authors:
  - name: Sagar Narwade
    orcid: 0009-0004-9636-3611
    affiliation: 1
  - name: Gillian Kant
    orcid: 0000-0003-2346-2841
    affiliation: 2
  - name: Benjamin Säfken
    orcid: 0000-0003-4702-3333
    affiliation: 1
  - name: Benjamin Leiding
    orcid:
    affiliation: 1

affiliations:
 - name: Technische Universität Clausthal, Clausthal-Zellerfeld, Germany
   index: 1
 - name: Georg-August-Universität Göttingen, Göttingen, Germany
   index: 2
date: 15 September 2023
bibliography: paper.bib

---

# Summary

SIRITVIS represents a powerful text analysis tool meticulously engineered for the purpose of parsing Reddit, Twitter, and Instagram data. It harnesses the capabilities of sophisticated Topic Models to enable unsupervised information extraction, thereby streamlining the processes of data collection, cleansing, and model optimization. A notable highlight of SIRITVIS lies in its ability to conduct geospatial mapping of social media posts on a global scale, forging connections between geographical locations and prevalent trending topics. Additionally, it offers sentiment analysis through the acclaimed NLTK VADER tool. This software stands as an invaluable asset for the scientific community, affording in-depth insights into public discourse across diverse social platforms. Its utility spans across a spectrum of research objectives, with potential applications including the analysis of global blockchain discussions. Installation is a straightforward endeavor via the pip package manager, complemented by comprehensive installation instructions readily available in the package's dedicated repository ^[https://github.com/CodeEagle22/SIRITVIS/].


## Introduction

The rise of social media platforms has transformed the way we communicate, share information, and express opinions on a wide range of topics. Reddit and Twitter, as two of the most popular platforms, have become significant sources of public discourse. Analyzing text data from these platforms can provide valuable insights into public sentiments, preferences, and trending discussions, benefiting fields such as marketing, politics, and disaster management.

However, handling the massive volume of unstructured text data from social media can be challenging due to its dynamic nature and sheer size. To address this challenge, we introduce SIRITVIS, a text analysis package designed to simplify the analysis of Reddit and Twitter data. SIRITVIS uses advanced Topic Models developed by AVITM [@srivastava2017autoencoding], which include Latent Dirichlet Allocation (LDA), Neural Latent Dirichlet Allocation (NeuralLDA), Prod Latent Dirichlet Allocation (ProdLDA), and CTM Topic Models, to automatically identify and extract topics in an unsupervised manner. These models allow users to explore large text datasets and discover hidden patterns for meaningful insights.

SIRITVIS offers a range of features to streamline the entire data preparation process, from data collection to model evaluation. Users can easily collect Reddit posts and Tweets from global locations using developer accounts. The package includes efficient data cleaning, transformation, training, and evaluation functionalities, ensuring that the data is well-prepared for topic model analysis. To address sparse social media data, SIRITVIS employs hashtag pooling, a technique to improve result quality.

One notable feature of SIRITVIS is its ability to map the spatial distribution of Tweets on a world map, associating each location with its top trending topics and their frequency. Additionally, it classifies and color-codes locations based on the sentiments expressed in tweets, providing counts of positive, negative, and neutral tweets. Users can also easily explore specific keywords and visualize their occurrences on the world map. This spatial insight enhances our understanding of public discussions and supports data-driven decision-making across various domains.

SIRITVIS bridges the gap between text data and spatial insights, making it a valuable tool for scientific research and decision-making. In this paper, we introduce the capabilities of SIRITVIS and demonstrate its potential through real-world examples. Furthermore, we discuss how SIRITVIS distinguishes itself by combining advanced Topic Models with spatial visualization, sentiment analysis, and automated parameter optimization. This innovative tool holds promise in unlocking hidden knowledge in social media data and facilitating data-driven decision-making across diverse domains.


## Comparing and Contrasting Available Toolsets

SIRITVIS distinguishes itself prominently among existing toolkits designed for the analysis of text data from social media due to its exceptional versatility. While alternatives such as TTLocVis [@Kant2020] and TweetViz [@stojanovski2014] undoubtedly offer value, SIRITVIS stands out by virtue of its rich array of advanced topic models, a distinctive tweet mapping functionality for geospatial insights, and seamless integration with "pyLDAvis" for the enhancement of result visualization.

What truly sets SIRITVIS apart is its comprehensive suite of evaluation metrics using the octis tool [@terragni2020octis], which encompasses critical aspects such as topic diversity, accuracy, inverted RBO, coherence, and Jaccard similarity. This robust evaluation framework serves as a hallmark, ensuring the creation of topic models that are not only reliable but also imbued with substantive meaning.

Through its multifaceted approach, SIRITVIS not only delivers profound and comprehensive analysis but also offers a broad spectrum of capabilities. Consequently, it has emerged as an indispensable tool for extracting actionable insights from text data originating on social media platforms.




# Figures

![Topic Mapper.\label{fig:Topic Mapper}](topic_mapper.png){ width=80% }

# References