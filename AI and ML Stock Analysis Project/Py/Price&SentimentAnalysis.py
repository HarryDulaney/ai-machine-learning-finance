"""
  The first section of this program  predicts the price of stocks using a machine learning
  technique called Long Short-Term Memory (LSTM).

  LSTMs (“long short-term memory” units) are the most powerful and well known subset of Recurrent Neural Networks.
  They are a type of artificial neural network designed to recognize patterns in sequences of data, such as numerical
  times series data.
  
  DISCLAIMER: A core characteristic of financial markets is unpredictability. 
  Historic prices' influence on future value is often a result of market makers relying on past price trend to make buying and selling decisions 
  It is by no means a definite feature of market behavior and is NOT guaranteed to predict the future price of a security. 
  Trend analysis is one tool in an investors toolbox and should be used alongside fundamental analysis and diversification for making sound investing decisions.
"""

import math
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as web
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from urllib.request import urlopen
import pandas as pd
from bs4 import BeautifulSoup
import nltk
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
from collections import OrderedDict

data = web.DataReader('MSFT', data_source='yahoo', start='2012-01-01', end='2020-04-20')

"""
The stock price of today will depend upon:

1.   The trend that the stock has been following in the previous days
2.   Forecasting the price of a firm’s stock, investors should use not just past prices of the stock but also any other
     information that is helpful in forecasting the future profitability of the firm, including the quality of the firm’s 
     management, new products the firm might be developing, and so on

Component in Model
1.   The previous cell state (i.e. the information that was present in the memory after the previous time step)

2.   The previous hidden state (i.e. this is the same as the output of the previous cell)

3.   The input at the current time step (i.e. the new information that is being fed in at that moment)
"""

plt.style.use('fivethirtyeight')

plt.figure(figsize=(20, 10))
plt.title('Closing Prices for MSFT')
plt.plot(data['Close'])
plt.xlabel('Year', fontsize=20)
plt.ylabel('Closing Price', fontsize=20)
plt.show()

data_new = data.filter(['Close'])
dataset = data_new.values
training_length = math.ceil(len(dataset) * .8)

# Scale the data set to be values between 0 and 1

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)

# Create a training data set that contains the past 60 day closing price values
# and split the data into x_train and y_train data sets

train = scaled[0:training_length, :]
X_train = []
Y_train = []
for i in range(60, len(train)):
    X_train.append(train[i - 60:i, 0])
    Y_train.append(train[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model to have two LSTM layers with 50 neurons and
# two Dense layers, one with 25 neurons and the other with 1 neuron

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

""" 
Mean squared error is calculated as the average of the squared differences between the predicted and actual values.
 The result is always positive regardless of the sign of the predicted and actual values.
 The squaring means that larger mistakes result in more error than smaller mistakes, 
 meaning that the model is punished for making larger mistakes.
 """

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, batch_size=1, epochs=1)

test = scaled[training_length - 60:, :]

X_test = []
Y_test = dataset[training_length:, :]
for i in range(60, len(test)):
    X_test.append(test[i - 60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predicate values from the model using the test data.

predict = model.predict(X_test)
predict = scaler.inverse_transform(predict)

train_new = data_new[:training_length]
valid = data_new[training_length:]
valid['predict'] = predict

plt.figure(figsize=(20, 10))
plt.title('Predicted Stock Price vs. Actual Stock Price (MSFT)')
plt.xlabel('Year', fontsize=20)
plt.ylabel('Closing Price', fontsize=20)
plt.plot(train_new['Close'])
plt.plot(valid[['Close', 'predict']])
plt.legend(['Trained Price', 'Actual Price', 'Predicted Price'], loc='top left')
plt.show()

# **_Sentiment Analysis_**
# To further forecast how advantageous an investment in Microsoft stock will be,
# we can utilize Sentiment Analysis.

# We begin by updating Vader's default Lexicon with Loughran and McDonald's positive and negative word lists.
# The_ **Lexicon** _is a large date-store of words and characters and the degree
# to which they describe the sentiment of a written message_
# **Vader's Lexicon**  is customized and tuned to analyze social media messages.
# So, we are going to update it with a lexicon that's specifically tailored to analysing **financial sentiment**.

analyzer = SentimentIntensityAnalyzer()

# Begin updating VADER's default Lexicon with Loughran and McDonald words lists
# Loughran and McDonald words list for positive sentiment
# For Local Deploy
# url_pos = ('datasource/LoughranMcDonald_positive_word_lists.csv')
# url_neg = ('datasource/LoughranMcDonald_negative_word_lists.csv')

# Read in the Loughran and McDonald words list for positive sentiment
# For web deploy
url_pos = (
    'https://raw.githubusercontent.com/HarryDulaney/ai-machine-learning-finance/master/LoughranMcDonald_positive_word_lists.csv')

# Read in the Loughran and McDonald words list for negative sentiment
url_neg = (
    'https://raw.githubusercontent.com/HarryDulaney/ai-machine-learning-finance/master/LoughranMcDonald_negative_word_lists.csv')

colnames = ['word']

positive_words = pd.read_csv(url_pos, names=colnames, header=None, delim_whitespace=True)
negative_words = pd.read_csv(url_neg, names=colnames, header=None, delim_whitespace=True)

positive = positive_words['word'].to_list()
negative = negative_words['word'].to_list()

custom_lexicon = {}
custom_lexicon.update({word: 2.0 for word in positive})
custom_lexicon.update({word: -2.0 for word in negative})
custom_lexicon.update(analyzer.lexicon)
analyzer.lexicon = custom_lexicon
# End VADER Lexicon customization


# Dictionary to hold score and date results

scores = {}

# Iterate through search results page.
# Minimum value is "(1,'2')" this will go back to late Feb
# For most useful analysis with a still reasonable runtime "range(1,10)"
for i in range(1, 10):
    page = urlopen('https://www.businesstimes.com.sg/search/microsoft?page=' + str(i)).read()
    soup = BeautifulSoup(page, features="html.parser")
    # Find the html tag matching <div class="media-body">
    # This is the Headline, Paragraph, and Link for search results aka 'post'
    posts = soup.findAll("div", {"class": "media-body"})
    # Iterate over each post, open link to article, use Bsoup to find html paragraph tags, read and analyze the text
    for post in posts:
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

ordered_scores = OrderedDict(scores)  # Convert scores to an Ordered(Sorted) Dictionary

# Reverse order to chart earliest date to latest date
sort_list = collections.OrderedDict(reversed(list(ordered_scores.items())))
x = []
y = []
# Unpack paired list into separate lists
for key, value in sort_list.items():
    x.append(key)
    y.append(value)

# Vader Legend -> score >= 0.05 == positive else if -> score <= -0.05 == negative
# Plot
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20, 10))
plt.title('Changing sentiment of MSFT over time')
plt.axhline(y=-0.05, color='r', linestyle='-')
plt.axhline(y=0.05, color='b', linestyle='-')
plt.text(0, 0.06, 'Positive Range Start')
plt.text(0, -0.03, 'Negative Range Start')
plt.plot(x, y)
plt.show()
