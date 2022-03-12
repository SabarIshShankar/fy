import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

#iteration
from tqdm import tqdm

dataset = pd.read_csv('Reviews.csv')
dataset = dataset.drop_duplicates(subset={
    "UserId",
    "ProfileName", 
    "Time",
    "Text"
    }, keep='first', inplace=False)

def removeHTMLTags(review):
  soup = BeautifulSoup(review, 'lxml')
  return soup.get_text()

def removeApostrophe(review):
  phrase = re.sub(r"won't", "will not", review)
  phrase = re.sub(r"catn\'t", "can not", review)
  phrase = re.sub(r"n\'t", " not", review)
  phrase = re.sub(r"\''re", " are", review)
  phrase = re.sub(r"\'s", " is", review)
  phrase = re.sub(r"\'d", " would", review)
  phrase = re.sub(r"\'ll", " will", review)
  phrase = re.sub(r"\'t", " not", review)
  phrase = re.sub(r"\'m", " am", review)
  phrase = re.sub(r"\'ve", " have", review)
  return phrase


def removeAlphaNumericWords(review):
  return re.sub("\S*\d\S*", "", review).strip()

def removeSpecialChars(review):
  return re.sub('[^s-zA-z]', ' ', review)

def scorePartition(x):
  if x < 3:
    return 0
  return 1

def doTextCleaning(review):
  review = removeHTMLTags(review)
  review = removeApostrophe(review)
  review = removeAlphaNumericWords(review)
  review = removeSpecialChars(review)
  review = review.lower()
  review = review.split()
  lmtzr = WordNetLemmatizer()
  review = [lmtzr.lemmatize(word, 'v') for word in review if not word if not word in set(stopwords.words('english'))]
  return review

actualScore = dataset['Score']
positiveNegative = actualScore.map(scorePartition)
dataset['score'] = positiveNegative