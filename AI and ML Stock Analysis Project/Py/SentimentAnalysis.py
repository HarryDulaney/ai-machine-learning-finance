import time
from urllib.request import urlopen
import requests
import pandas as pd
import csv
import numpy as np
import keras as ker
from bs4 import BeautifulSoup
import nltk
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

# Begin update VADER's default Lexicon with Loughran and McDonald words lists and preconfigured stock lexicon
# Read in the Loughran and McDonald words list for positive sentiment
# For web deploy
url_pos = (
    'https://raw.githubusercontent.com/HarryDulaney/ai-machine-learning-finance/master/LoughranMcDonald_positive_word_lists.csv')

# Read in the Loughran and McDonald words list for negative sentiment
url_neg = (
    'https://raw.githubusercontent.com/HarryDulaney/ai-machine-learning-finance/master/LoughranMcDonald_negative_word_lists.csv')

# For Local Deploy
# url_pos = ('datasource/LoughranMcDonald_positive_word_lists.csv')
# url_neg = ('datasource/LoughranMcDonald_negative_word_lists.csv')


positive = []
with open(url_pos, 'r') as pos:
    reader = csv.reader(pos)
    for row in reader:
        positive.append(row[0].strip())

negative = []
with open(url_neg, 'r') as neg:
    reader = csv.reader(neg)
    for row in reader:
        entry = row[0].strip().split(" ")
        if len(entry) > 1:
            negative.extend(entry)
        else:
            negative.append(entry[0])

custom_lexicon = {}
custom_lexicon.update({word: 2.0 for word in positive})
custom_lexicon.update({word: -2.0 for word in negative})
custom_lexicon.update(analyzer.lexicon)
analyzer.lexicon = custom_lexicon

# End VADER Lexicon customization

scores = {}

# Iterate through search results pages
for i in range(1, 2):
    page = urlopen('https://www.businesstimes.com.sg/search/microsoft?page=' + str(i)).read()
    soup = BeautifulSoup(page, features="html.parser")
    # Find the html tag matching <div class="media-body">
    # This is the Headline, Paragraph, and Link for search results aka 'post'
    posts = soup.findAll("div", {"class": "media-body"})
    # Iterate over each post, open link to article, use Bsoup to find html paragraph tags, read and analyze the text
    for post in posts:
        time.sleep(1)
        url = post.a['href']
        date = post.time.text
        print(date, url)
        try:
            link_page = urlopen(url).read()
        except:
            url = url[:-2]
            link_page = urlopen(url).read()
        link_soup = BeautifulSoup(link_page)
        sentences = link_soup.findAll("p")
        passage = ""
        for sentence in sentences:
            passage += sentence.text
        sentiment = analyzer.polarity_scores(passage)['compound']
        scores.setdefault(date, []).append(sentiment)

plt.figure(figsize=(20, 10))
plt.title('Changing sentiment of MSFT over time')
plt.xlabel('Date', fontsize=20)
plt.ylabel('Polarity Score - Vader expanded lexicon to include Loughran & McDonald words ', fontsize=15)
plt.plot(scores.values().__getattribute__('sentiment'))
plt.plot(valid[['Close', 'predict']])
plt.legend(['Trained Price', 'Actual Price', 'Predicted Price'], loc='top left')
plt.show()
