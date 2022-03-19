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

def hashtag_extract(x):
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

HT_regular = hashtag_extract(train['tweet'][train['label'] == 0])
HT_negative = hashtag_extract(train['tweet'][train['label'] == 1])
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

import nltk
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()), 'Count': list(a.values())})
d = d.nlargest(columns = "Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=d, x = "Hashtag", y = "Count")
ax.set(ylabel="Count")
plt.show()

a = nltk.FreqDist(HT_negative)
d = pd.DataFrame({'Hashtag':list(a.keys()), 'Count': list(a.values())})
d = d.nlargest(columns="Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=d, x="Hashtag", y ="Count")
ax.set(ylabel="Count")
plt.show()

import gensim #generate similar
tokenized_tweet =  train['tweet'].apply(lambda x:x.split()) #break up the strings
model_w2v = gensim.models.Word2Vec(tokenized_tweet, size=200,
                                   window=5,
                                   min_count=2,
                                   sg=1,
                                   hs=0,
                                   negative=10,
                                   workers=2,
                                   seed=34)
model_w2v.train(tokenized_tweet, total_examples = len(train['tweet']), epochs=20)

model_w2v.wv.most_similar(positive="dinner")

model_w2v.wv.most_similar(positive="apple")

model_w2v.wv.most_similar(positive="law")

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models.doc2vec import LabeledSentence

def add_label(twt):
  output=[]
  for i, s in zip(twt.index, twt):
    output.append(LabeledSentence(s, ["tweet_" + str(i)]))
  return output

labeled_tweets = add_label(tokenized_tweet)
labeled_tweets[:6]

nltk.download('stopwords') #stopwords
from nltk.corpus import stopwords #corpus reads the text
from nltk.stem.porter import PorterStemmer
#porter stemmer reduces different words into one

train_corpus = []
#total number of social text
for i in range(0, 31962): 
  review = re.sub('[^a-zA-Z]', '', train['tweet'][i])
  review = review.lower()
  review = review.split()

  ps = PorterStemmer()
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

  review = ''.join(review)
  train_corpus.append(review)

test_corpus = []
for i in range(0, 17197):
  review = re.sub('[^a-zA-Z]', '', test['tweet'][i])
  review =  review.lower()
  review = review.split()

  ps = PorterStemmer()
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  review = ''.join(review)
  test_corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
#scikit learn used for feature extaction (reducing the number)
#and count vectorizer is used to convert the text into vectors which acts like a list
cv = CountVectorizer(max_features = 2500)
x = cv.fit_transform(train_corpus).toarray()
y = train.iloc[:, 1]

print(x.shape)
print(y.shape)