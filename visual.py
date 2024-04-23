
#########################################################
# VISUALIZACIÓN
#########################################################
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# NLTK libraries
import nltk
import re
import string
import plotly.express as px

from jedi.api.refactoring import inline
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
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
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from textblob import TextBlob
from plotly import tools, subplots
import plotly.graph_objs as go
from plotly.offline import iplot
from sklearn.feature_selection import SequentialFeatureSelector

# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

# Other miscellaneous libraries
from scipy import interp
from itertools import cycle
from collections import defaultdict
from collections import Counter
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
from opencage.geocoder import OpenCageGeocode
import geocoder
from geopy.geocoders import Nominatim
from geopy import distance
from geopy.distance import great_circle, geodesic
from scipy.stats import pearsonr


stop_words = ['yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each',
              'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
              'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above',
              'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't",
              'very', 'should', 'any', 'y', 'isn', 'who', 'a', 'they', 'to', 'too', "should've", 'has', 'before',
              'into', 'yours', "it's", 'do', 'against', 'on', 'now', 'her', 've', 'd', 'by', 'am', 'from',
              'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
              'his', 'himself', 'ourselves', 'was', 'through', 'out', 'below', 'own', 'myself', 'theirs',
              'me', 'why', 'once', 'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
              'at', 'after', 'its', 'which', 'there', 'our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
              'over', 'again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all', 'flight', 'plane',
              'singapore',
              'airlines', 'airline', 'turkish']

corrections_dict = {
    "Zurich via Singapore": "Zurich",
    "Siam Reap, Cambodia": "Siem Reap",
    "Singaporw": "Singapore",
    "Singaporec": "Singapore",
    "Singapore Return": "Singapore",
    "London Heahrow": "London Heathrow",
    "Singapoe": "Singapore",
    "Sinhapore": "Singapore",
    "Qingdoa": "Qingdao",
}


#############################################################
API_KEY = "6195db615c5f44deabbd7c18adddaf7d"

stop_words = ['yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each',
              'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
              'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above',
              'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't",
              'very', 'should', 'any', 'y', 'isn', 'who', 'a', 'they', 'to', 'too', "should've", 'has', 'before',
              'into', 'yours', "it's", 'do', 'against', 'on', 'now', 'her', 've', 'd', 'by', 'am', 'from',
              'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
              'his', 'himself', 'ourselves', 'was', 'through', 'out', 'below', 'own', 'myself', 'theirs',
              'me', 'why', 'once', 'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
              'at', 'after', 'its', 'which', 'there', 'our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
              'over', 'again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all', 'flight', 'plane',
              'singapore',
              'airlines', 'airline', 'turkish']

corrections_dict = {
    "Zurich via Singapore": "Zurich",
    "Siam Reap, Cambodia": "Siem Reap",
    "Singaporw": "Singapore",
    "Singaporec": "Singapore",
    "Singapore Return": "Singapore",
    "London Heahrow": "London Heathrow",
    "Singapoe": "Singapore",
    "Sinhapore": "Singapore",
    "Qingdoa": "Qingdao",
}


###############################################################
# FUNCIONES
###############################################################

def sent(rating):
    if rating["Overall Rating"] >= 7:
        value = "Positive"
    elif rating["Overall Rating"] == 5 or (rating["Overall Rating"]) == 6:
        value = "Neutral"
    else:
        value = "Negative"
    return value


def get_origin(route):
    return route.split(' to ')[0]


def get_destiny(route):
    if ' to ' in route:
        to_parts = route.split(' to ')
        if ' via ' in to_parts[1]:
            return to_parts[1].split(' via ')[0]
        else:
            return to_parts[1]
    else:
        return 'None'


def get_scale(route):
    if ' via ' in route:
        return route.split(' via ')[1]
    else:
        return 'None'


def bool_scale(route):
    if 'None' in route:
        return False
    else:
        return True


def clean_review(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation='h',
        marker=dict(
            color=color,
        ),
    )
    return trace


def get_lat(route):
    geocoder = OpenCageGeocode(API_KEY)
    lat = None
    query = route
    results = geocoder.geocode(query)
    if len(results) > 0:
        lat = results[0]['geometry']['lat']
    return lat


def get_long(route):
    geocoder = OpenCageGeocode(API_KEY)
    query = route
    results = geocoder.geocode(query)
    lng = results[0]['geometry']['lng']
    return lng


def get_country(route):
    geocoder = OpenCageGeocode(API_KEY)
    query = route
    results = geocoder.geocode(query)
    country = results[0]['components']['country']
    return country


def read_country(city):
    """
    Convert cities and returns the country
    """
    geolocator = Nominatim(user_agent="google")  # user agent can be any user agent
    location = geolocator.geocode(city,
                                  language="en")  # specified the language as some countries are in other lanaguages

    if location is None:
        return None

    if hasattr(location, 'address'):
        country = location.address.split(',')[
            -1]  # split the string based on comma and retruns the last element (country)
        continent = location.address.split(',')[0].strip()
    latitude = location.latitude
    longitude = location.longitude
    location_info = {
        "country": country,
        "continent": continent,
        "latitude": latitude,
        "longitude": longitude
    }
    return location_info



reviews = pd.read_csv("airlines_reviewsSingapore.csv")
print("Singapore Airline Dataset:")
print(reviews.head(5))
reviews.drop(
    ['Seat Comfort', 'Staff Service', 'Food & Beverages', 'Inflight Entertainment', 'Value For Money', 'Airline',
     'Recommended'], axis=1,
    inplace=True)
reviews["Sentiment"] = reviews.apply(sent, axis=1)
reviews.drop("Overall Rating", axis=1, inplace=True)
reviews["Origin"] = reviews["Route"].apply(get_origin)
reviews["Destiny"] = reviews["Route"].apply(get_destiny)
reviews["Origin"] = reviews["Origin"].replace(corrections_dict)
reviews["Destiny"] = reviews["Destiny"].replace(corrections_dict)
reviews["Scale"] = reviews["Route"].apply(get_scale)
reviews["Scale_bool"] = reviews["Scale"].apply(bool_scale)

# Splitting the date
re_new = reviews["Review Date"].str.split("-", n=2, expand=True)

# adding month to the main dataset
reviews["year"] = re_new[0]

# adding day to the main dataset
reviews["month"] = re_new[1]

# adding day to the main dataset
reviews["day"] = re_new[2]
print(reviews.head())

reviews['Rev'] = reviews['Reviews'] + reviews['Title']
reviews = reviews.drop(['Reviews', 'Title'], axis=1)
print(reviews.head())

# Simplificar texto según función clean_review
reviews['Reviews_Simp'] = reviews["Rev"].apply(lambda x: clean_review(x))
print(reviews.head())

# Quitar stop words
reviews['Reviews_Simp'] = reviews['Reviews_Simp'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
print(reviews.head())

reviews['polarity'] = reviews['Reviews_Simp'].map(lambda text: TextBlob(text).sentiment.polarity)
reviews['review_len'] = reviews['Reviews_Simp'].astype(str).apply(len)
reviews['word_count'] = reviews['Reviews_Simp'].apply(lambda x: len(str(x).split()))
print(reviews.head())


plt.figure(figsize=(6, 6))
plt.title('Percentage of Sentiments')
ax = sns.countplot(y="Sentiment", data=reviews)
total = len(reviews)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_width() / total)
    x = p.get_x() + p.get_width() + 0.02
    y = p.get_y() + p.get_height() / 2
    ax.annotate(percentage, (x, y))

plt.show()

reviews.groupby(['year', 'Sentiment'])['Sentiment'].count().unstack().plot(legend=True)
plt.title('Year and Sentiment count')
plt.xlabel('Year')
plt.ylabel('Sentiment count')
plt.show()

# Creating a dataframe
dayreview = pd.DataFrame(reviews.groupby('day')['Reviews_Simp'].count()).reset_index()
dayreview['day'] = dayreview['day'].astype('int64')
dayreview.sort_values(by=['day'])

# Plotting the graph
sns.barplot(x="day", y="Reviews_Simp", data=dayreview)
plt.title('Day vs Reviews count')
plt.xlabel('Day')
plt.ylabel('Reviews count')
plt.show()


polarity_df = pd.DataFrame(reviews['polarity'], columns=['polarity'])

# Crear el histograma utilizando Plotly
fig = go.Figure()

fig.add_trace(go.Histogram(x=polarity_df['polarity'], nbinsx=50, marker_color='skyblue'))

fig.update_layout(
    title='Sentiment Polarity Distribution',
    xaxis_title='Polarity',
    yaxis_title='Count',
    bargap=0.05,
    template='plotly_white'
)
fig.show()

# Review lenght

review_len_df = pd.DataFrame(reviews['review_len'], columns=['review_len'])

# Crear el histograma utilizando Plotly
fig = go.Figure()

fig.add_trace(go.Histogram(x=review_len_df['review_len'], nbinsx=150, marker_color='green'))

fig.update_layout(
    title='Review Length Distribution',
    xaxis_title='Review Length',
    yaxis_title='Count',
    bargap=0.05,
    template='plotly_white'
)

fig.show()
# Word count

word_count_df = pd.DataFrame(reviews['word_count'], columns=['word_count'])

# Crear el histograma utilizando Plotly
fig = go.Figure()

fig.add_trace(go.Histogram(x=word_count_df['word_count'], nbinsx=150, marker_color='pink'))

fig.update_layout(
    title='Word Count Distribution',
    xaxis_title='Word count',
    yaxis_title='Count',
    bargap=0.05,
    template='plotly_white'
)

fig.show()



# Filtering data
#######################################################################3
# SEPARAR POSITIVOS/NEGATIVOS/NEUTRALES
#######################################################################
positive_review = reviews[reviews["Sentiment"] == 'Positive'].dropna()
neutral_review = reviews[reviews["Sentiment"] == 'Neutral'].dropna()
negative_review = reviews[reviews["Sentiment"] == 'Negative'].dropna()


## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]


print("Negative Reviews: ")
print(negative_review.head())

## Get the bar chart from positive reviews ##
freq_dict = defaultdict(int)
for sent in positive_review["Reviews_Simp"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(20), 'blue')

## Get the bar chart from neutral reviews ##
freq_dict = defaultdict(int)
for sent in neutral_review["Reviews_Simp"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(20), 'purple')

## Get the bar chart from negative reviews ##
freq_dict = defaultdict(int)
for sent in negative_review["Reviews_Simp"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(20), 'yellow')

# Creating two subplots
fig = subplots.make_subplots(rows=3, cols=1, vertical_spacing=0.04,
                             subplot_titles=["Frequent words of positive reviews", "Frequent words of neutral reviews",
                                             "Frequent words of negative reviews"])
fig.add_trace(trace0, 1, 1)
fig.add_trace(trace1, 2, 1)
fig.add_trace(trace2, 3, 1)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
iplot(fig, filename='word-plots')

# BIGRAMA
## Get the bar chart from positive reviews ##
freq_dict = defaultdict(int)
for sent in positive_review["Reviews_Simp"]:
    for word in generate_ngrams(sent, 2):
        freq_dict[word] += 1
positive_review["bigrams"] = positive_review["Reviews_Simp"].apply(lambda sent: list(generate_ngrams(sent, 2)))
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(20), 'blue')

## Get the bar chart from neutral reviews ##
freq_dict = defaultdict(int)
for sent in neutral_review["Reviews_Simp"]:
    for word in generate_ngrams(sent, 2):
        freq_dict[word] += 1
neutral_review["bigrams"] = neutral_review["Reviews_Simp"].apply(lambda sent: list(generate_ngrams(sent, 2)))
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(20), 'purple')

## Get the bar chart from negative reviews ##
freq_dict = defaultdict(int)
for sent in negative_review["Reviews_Simp"]:
    for word in generate_ngrams(sent, 2):
        freq_dict[word] += 1
negative_review["bigrams"] = negative_review["Reviews_Simp"].apply(lambda sent: list(generate_ngrams(sent, 2)))
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(20), 'yellow')

# Creating two subplots
fig = subplots.make_subplots(rows=3, cols=1, vertical_spacing=0.04,
                             subplot_titles=["Frequent words of positive reviews", "Frequent words of neutral reviews",
                                             "Frequent words of negative reviews"])
fig.add_trace(trace0, 1, 1)
fig.add_trace(trace1, 2, 1)
fig.add_trace(trace2, 3, 1)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
iplot(fig, filename='word-plots')

# N = 3

## Get the bar chart from positive reviews ##
freq_dict = defaultdict(int)
for sent in positive_review["Reviews_Simp"]:
    for word in generate_ngrams(sent, 3):
        freq_dict[word] += 1
positive_review["threegram"] = positive_review["Reviews_Simp"].apply(lambda sent: list(generate_ngrams(sent, 3)))
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(20), 'blue')
positive_review.to_csv("turk_pos.csv", index=False)
fd_sorted.to_csv("sorted.csv", index=False)
## Get the bar chart from neutral reviews ##
freq_dict = defaultdict(int)
for sent in neutral_review["Reviews_Simp"]:
    for word in generate_ngrams(sent, 3):
        freq_dict[word] += 1
neutral_review["threegram"] = neutral_review["Reviews_Simp"].apply(lambda sent: list(generate_ngrams(sent, 3)))
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(20), 'purple')
neutral_review.to_csv("turk_neu.csv", index=False)

## Get the bar chart from negative reviews ##
freq_dict = defaultdict(int)
for sent in negative_review["Reviews_Simp"]:
    for word in generate_ngrams(sent, 3):
        freq_dict[word] += 1
negative_review["threegram"] = negative_review["Reviews_Simp"].apply(lambda sent: list(generate_ngrams(sent, 3)))
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(20), 'yellow')
negative_review.to_csv("turk_neg.csv", index=False)

# Creating two subplots
fig = subplots.make_subplots(rows=3, cols=1, vertical_spacing=0.04,
                             subplot_titles=["Frequent words of positive reviews", "Frequent words of neutral reviews",
                                             "Frequent words of negative reviews"])
fig.add_trace(trace0, 1, 1)
fig.add_trace(trace1, 2, 1)
fig.add_trace(trace2, 3, 1)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
iplot(fig, filename='word-plots')

# WORDCLOUD

text = positive_review["Reviews_Simp"]
wordcloud = WordCloud(
    width=3000,
    height=2000,
    background_color='black',
    stopwords=STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize=(40, 30),
    facecolor='k',
    edgecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

text = neutral_review["Reviews_Simp"]
wordcloud = WordCloud(
    width=3000,
    height=2000,
    background_color='black',
    stopwords=STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize=(40, 30),
    facecolor='k',
    edgecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


text = negative_review["Reviews_Simp"]
wordcloud = WordCloud(
    width=3000,
    height=2000,
    background_color='black',
    stopwords=STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize=(40, 30),
    facecolor='k',
    edgecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
