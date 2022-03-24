#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
trends = TrendReq()

trends.build_payload(kw_list=["machine learning"])
data = trends.interest_by_region()
data = data.sort_values(by="machine learning", ascending=False)
data = data.head(10)
print(data)


# In[17]:


data.reset_index().plot(x="geoName", y="machine learning", kind="bar")
plt.style.use('fivethirtyeight')
plt.show()


# In[19]:


data = TrendReq(hl='en-US', tz=360)
data.build_payload(kw_list=['machine learning'])
data = data.interest_over_time()
fig, ax = plt.subplots(figsize=(20, 15))
data['machine learning'].plot()
plt.style.use('fivethirtyeight')
plt.title('Tootal google searches for machine learning')
plt.show()


# In[26]:


import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re
import unicodedata
import nltk
import json
import inflect
import matplotlib.pyplot as plt

