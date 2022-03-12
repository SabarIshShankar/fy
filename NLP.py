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