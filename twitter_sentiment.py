import twitter
import csv
import time   
import pandas as pd
import os
from pandas import DataFrame
import numpy as np
import collections
from sklearn.model_selection import train_test_split
import random
import re
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 

# # pip install nltk
# # python -m nltk.downloader stopwords

# # Go into a python shell and type:
# # >>> import nltk
# # >>> nltk.download()
# # Then an installation window appears. Go to the 'Models' tab and select 'punkt' 
# # from under the 'Identifier' column. Then click Download and it will install the necessary files. 
# # Then it should work!


# initialize api instance
twitter_api = twitter.Api(consumer_key='cOT89KTGWyV7N5oJ2vneOidDr',
                        consumer_secret='2srz3k9OT1xQaMd00m5dFykRKimboRxBf5ARBntxhUkiv1Hlaa',
                        access_token_key='1295166185961476097-Vjbz2sw46i6VlF7IueJPzWkyjbMtJo',
                        access_token_secret='HnEi7O5TlqiFUQP7gNeT8Sct96P16JE7wbX1iKmacVBeC')

# test authentication
# print(twitter_api.VerifyCredentials())

# Import the csv file
script_dir = os.getcwd() + '/ml/datasets/'
file = 'tweets.csv'
tweetDataFile = os.path.normcase(os.path.join(script_dir, file))

# Read from CSV to Pandas DataFrame
df = pd.read_csv(os.path.normcase(os.path.join(script_dir, file)), header=0, index_col=None)

# shuffle the DataFrame rows 
df = df.sample(frac = 1)

X = df[['text', 'label']]
y = df['label']
print(X)
print(X.shape)
print('#####')
print(y)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

class_counts = dict(collections.Counter(y))
print (f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print (f"X_test: {X_test.shape}, y_test: {y_test.shape}")
print (f"Classes: {class_counts}")

# get all english words
eng_words = set(nltk.corpus.words.words())

# stopwords package, punctuations, some constants, you can add your own words too
stopwords = set(stopwords.words('english') 
                    + list(punctuation) 
                    + ['url', 'google', 'microsoft' + 'apple' + 'twitter', 'rt', 'android'])

def prepare_data_set(data):
    datasets=[]
    for row in data.values:
        tweet_words = clean_tweet(row[0])
        label = row[1]
        # important point to be noted here, we are sending array of data, not individual elements
        # ((tweet_words,label)) instead of (tweet_words,label)
        # link words with classification label to train model
        datasets.append((tweet_words,label))
    return datasets
    
def clean_tweet(tweet):
    tweet = tweet.lower() 
    
    # find words starting from # like hashtags, so as to exclude them later in the code
    hashWords = set({tag.strip("#") for tag in tweet.split() if tag.startswith("#")})

    # find words starting from @, it is for usernames, so as to exclude them later in the code
    usernames = set({tag.strip("@") for tag in tweet.split() if tag.startswith("@")})

    # remove url starting from www or https
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    
    # create a dictionary of words from tweet
    tweet = word_tokenize(tweet)

    # exclude hashWords, usernames
    tweet = [word for word in tweet if word not in hashWords]
    tweet = [word for word in tweet if word not in usernames]

    # remove stopwords, declared above
    tweet = [word for word in tweet if word not in stopwords]

    # only allow english words, non-english words are not used for classification
    tweet = [word for word in tweet if word in eng_words]

    # remove words of length less than 2, like 'an', 'so'
    tweet = [word for word in tweet if len(word) > 2]

    # remove any word that contains non-alphabetical characters like '...'
    tweet = [word for word in tweet if word.isalpha() == True]

    return tweet

# prepare training set for X_train
trainingSet = prepare_data_set(X_train)

# prepare training set for X_test
testSet = prepare_data_set(X_test)

# make a list of all words in dataset
all_words = [] 
for data in trainingSet:
    all_words.extend(data[0])

# it creates mapping, how many times words appear, like 'new':198, 'galaxy':130
wordlist = nltk.FreqDist(all_words)

# it will create a keys of unique words, called word_features
word_features = wordlist.keys()

def convert_tweet_to_feature(tweet):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features[word]=(word in tweet_words)
    return features
    
# prepare training features 
trainingFeatures = []
for tweet in trainingSet:
    features = convert_tweet_to_feature(tweet[0])
    trainingFeatures.append((features, tweet[1])) 

# train the classifier
classifier=nltk.NaiveBayesClassifier.train(trainingFeatures)

# since model is ready now, use it on test dataset
testLabels = []
for tweet in testSet:
    features = convert_tweet_to_feature(tweet[0])
    label = classifier.classify(features)
    testLabels.append(label)

# compare actual test labels with our generated test labels from model, to see how accurate our model is
def find_accuracy(labels):
    good = 0
    bad = 0
    index = 0
    for label in labels:
        test_data = testSet[index][1]
        if (label == test_data):
            good = good + 1
        else:
            bad = bad + 1
        index = index+1
    return good/(good+bad)

print('Accuracy is', find_accuracy(testLabels))

# now, check on twitter real data

search_term = 'love'
real_tweets = twitter_api.GetSearch(search_term, count = 100)            
tweets = [[status.text, ''] for status in real_tweets]

dfTweets = DataFrame(tweets,columns=['text','label'])
newTweets = prepare_data_set(dfTweets)

# use classifier to guess labels using above model
twtLabels = [classifier.classify(convert_tweet_to_feature(tweet[0])) for tweet in newTweets]

# it shows five most features, which are used to label new tweets
classifier.show_most_informative_features(5)

def get_data_for_pie_chart():
    pos = 0
    neg = 0
    neu = 0
    irr = 0
    for label in twtLabels:
        if (label == 'positive'):
            pos = pos + 1 
        elif (label == 'negative'):
            neg = neg + 1
        elif (label == 'neutral'):
            neu = neu + 1
        elif (label == 'irrelevant'):
            irr = irr + 1

    sum = pos + neg + neu + irr;
    pos = pos/sum;
    neg = neg/sum;
    neu = neu/sum;
    irr = irr/sum;
    return pos, neg, neu, irr

import matplotlib.pyplot as plt
tweetsCategory = ["positive","negative","neutral", "irrelevant"]
       
Usage = [get_data_for_pie_chart()]

# Pie chart is oval by default
# To make it a circle
plt.axis("equal")

plt.pie(Usage,labels=tweetsCategory,autopct='%1.2f%%', shadow=False)
       
plt.title("Tweets Sentiments Analysis")
plt.show()