import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

train = pd.read_csv('train_tweet.csv')
test = pd.read_csv('test_tweets.csv')

print(train.shape)
print(test.shape)

test.head()

train.head()

train.isnull().any()
test.isnull().any()

train[train['label'] == 0].head(10)

train[train['label'] == 1].head(10)

train['label'].value_counts().plot.bar(figsize = (6, 4))

length_train = train['tweet'].str.len().plot.hist(color = "pink", figsize=(6, 4))
length_test = test['tweet'].str.len().plot.hist(color = 'blue', figsize = (6, 4))

train['len'] = train['tweet'].str.len()
test['len'] = test['tweet'].str.len()
train.head()
train.groupby('label').describe()

train.groupby('len').mean()['label'].plot.hist()
plt.title('variation of length')
plt.xlabel('Length')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = "english")
words = cv.fit_transform(train.tweet)
sum_words = words.sum(axis = 0)

words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x:x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
frequency.head(30).plot(x = 'word', y="freq", kind="bar", figsize=(15, 7))
plt.title("Frequent words")

from wordcloud import WordCloud
wordcloud = WordCloud().generate_from_frequencies(dict(words_freq))
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud)
plt.title('Vocabulary from reviews')

normal_words = ' '.join([text for text in train['tweet'][train['label'] == 0]])
wordcloud = WordCloud(width=800, height=500, random_state=0, max_font_size = 110).generate(normal_words)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Neutral words')
plt.show()

negative_words = ' '.join([text for text in train['tweet'][train['label'] == 1]])
wordcloud = WordCloud(random_state = 0).generate(negative_words)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.xasix('off')
plt.title('the negative words')
plt.show()

import re
def hashtag_extract(x):
  hashtags = []

  for i in x:
    ht = re.findall(r"#(\w+", i)
    hashtags.append(ht)

  return hashtags
