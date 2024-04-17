import matplotlib
import pandas as pd 
import ntlk

nltk.download('vader_lexicon')
nltk.download('movie_reviews')
nltk.download('punkt')

import textblob

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()

# Asking `SentimentIntensityAnalyzer` for the `polarity_score` gave us four values in a dictionary:

# **negative:** the negative sentiment in a sentence
# **neutral:** the neutral sentiment in a sentence
# **positive:** the postivie sentiment in the sentence
# **compound:** the aggregated sentiment. 

# sia.polarity_scores("This restaurant was great, but I'm not sure if I'll go there again.")

from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer

# blob = TextBlob("This restaurant was great, but I'm not sure if I'll go there again.")
# blob.sentiment

# blobber = Blobber(analyzer=NaiveBayesAnalyzer())

# def get_scores(content):
   # blob = TextBlob(content)
   # nb_blob = blobber(content)
   # sia_scores = sia.polarity_scores(content)
    
    # return pd.Series({
       # 'content': content,
       # 'textblob': blob.sentiment.polarity,
       # 'textblob_bayes': nb_blob.sentiment.p_pos - nb_blob.sentiment.p_neg,
       # 'nltk': sia_scores['compound'],
    # })

# scores = df.content.apply(get_scores)
# scores.style.background_gradient(cmap='RdYlGn', axis=None, low=0.4, high=0.4)