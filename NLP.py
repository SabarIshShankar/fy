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

corpus = []
for index, row in tqdm(dataset.iterrows()):
  review = doTextCleaning(row['Text'])
  corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(1, 3), max_features = 5000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,6].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, text_size = 0.20, random_state= 0)

#naivebayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#predicting results
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metric import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


def predictNewReview():
  newReview = input("Type the review")

  if newReview == '':
    print('Invalid Review')
  else:
    newReview = doTextCleaning(newReview)
    new_review = cv.transform([newReview]).toarray()
    prediction = classifier.predict(new_review)
    print(prediction)
    if prediction[0] == 1:
      print("Positive Review")
    else:
      print("Negative Review")