import matplotlib
import numpy as np
import pandas as pd

# NLTK libraries
import nltk
import re
import string

from jedi.api.refactoring import inline
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from imblearn.over_sampling import SMOTE
# Machine Learning libraries
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn import svm, datasets
from sklearn import preprocessing

# Metrics libraries
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from textblob import TextBlob
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import iplot

# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

# Other miscellaneous libraries
from scipy import interp
from itertools import cycle
from collections import defaultdict
from collections import Counter
from imblearn.over_sampling import SMOTE


stop_words= ['yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each',
             'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
             'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above',
             'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't",
             'very', 'should', 'any', 'y', 'isn', 'who',  'a', 'they', 'to', 'too', "should've", 'has', 'before',
             'into', 'yours', "it's", 'do', 'against', 'on',  'now', 'her', 've', 'd', 'by', 'am', 'from',
             'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
             'his', 'himself', 'ourselves',  'was', 'through', 'out', 'below', 'own', 'myself', 'theirs',
             'me', 'why', 'once',  'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
             'at', 'after', 'its', 'which', 'there','our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
             'over','again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all', 'flight', 'plane']

def sent(rating):
    if rating["Overall Rating"] >= 7:
        value = "Positive"
    elif rating["Overall Rating"] == 5 or (rating["Overall Rating"]) == 6:
        value = "Neutral"
    else:
        value = "Negative"
    return value


reviews = pd.read_csv("airlines_reviewsSingapore.csv")
print("Singapore Airline Dataset:")
print(reviews.head(5))
reviews.drop(['Seat Comfort', 'Staff Service', 'Food & Beverages', 'Inflight Entertainment', 'Value For Money'], axis=1,
             inplace=True)
reviews["Sentiment"] = reviews.apply(sent, axis=1)
print(reviews.head())

# reviews.rename(columns={"Overall Rating": "Sentiment"}, inplace=True)
## print shape of dataset with rows and columns and information
print("The shape of the  data is (row, column):" + str(reviews.shape))
print("The Information about the dataset:" + str(reviews.info()))

# Checking for null values
print('The null values in the dataset:')
print(reviews.isnull().sum())

reviews['Rev'] = reviews['Reviews'] + reviews['Title']
reviews = reviews.drop(['Reviews', 'Title'], axis=1)
print(reviews.head())

print("Count of sentiments:")
print(reviews["Sentiment"].value_counts())

# Splitting the date
re_new = reviews["Review Date"].str.split("-", n=2, expand=True)

# adding month to the main dataset
reviews["year"] = re_new[0]

# adding day to the main dataset
reviews["month"] = re_new[1]

# adding day to the main dataset
reviews["day"] = re_new[2]
print(reviews.head())

print('The year - wise count of reviews:')
print(reviews['year'].value_counts())

print('Year - wise count of sentiments:')
print(reviews.groupby(['year', 'Sentiment']).size())

print("Verified - wise count of sentiments")
print(reviews.groupby(['Verified', 'Sentiment']).size())

#print("Route - wise sentiments")
#print(reviews.groupby(['Route', 'Sentiment']).size())

print("Passenger - wise count of sentiments")
repeated_names = reviews['Name'].value_counts()[reviews['Name'].value_counts() > 1].index.tolist()

# Imprimir el conteo de sentimientos solo para los nombres repetidos
print("Passenger - wise count of sentiments for repeated names:")
print(reviews[reviews['Name'].isin(repeated_names)].groupby(['Name', 'Sentiment']).size())

# Verified or not?
def clean_review(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


reviews['Reviews_Simp'] = reviews["Rev"].apply(lambda x: clean_review(x))
print(reviews.head())

#stop words
reviews['Reviews_Simp'] = reviews['Reviews_Simp'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
print(reviews.head())

plt.figure(figsize=(6,6))
plt.title('Percentage of Sentiments')
ax = sns.countplot(y="Sentiment", data= reviews)
total = len(reviews)
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

plt.show()

reviews.groupby(['year','Sentiment'])['Sentiment'].count().unstack().plot(legend=True)
plt.title('Year and Sentiment count')
plt.xlabel('Year')
plt.ylabel('Sentiment count')
plt.show()
