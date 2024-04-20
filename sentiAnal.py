"""Referencias:
https://www.kaggle.com/code/soniaahlawat/sentiment-analysis-amazon-review"""

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
             'over','again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all', 'flight', 'plane', 'singapore',
             'airlines', 'airline']

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
reviews.drop(['Seat Comfort', 'Staff Service', 'Food & Beverages', 'Inflight Entertainment', 'Value For Money', 'Airline'], axis=1,
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

# custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y =df["word"].values[::-1],
        x = df["wordcount"].values[::-1],
        showlegend = False,
        orientation = 'h',
        marker = dict(
            color = color,
        ),
    )
    return trace

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

#plt.show()

reviews.groupby(['year','Sentiment'])['Sentiment'].count().unstack().plot(legend=True)
plt.title('Year and Sentiment count')
plt.xlabel('Year')
plt.ylabel('Sentiment count')
#plt.show()

#Creating a dataframe
dayreview = pd.DataFrame(reviews.groupby('day')['Reviews_Simp'].count()).reset_index()
dayreview['day'] = dayreview['day'].astype('int64')
dayreview.sort_values(by = ['day'])

#Plotting the graph
sns.barplot(x = "day", y = "Reviews_Simp", data = dayreview)
plt.title('Day vs Reviews count')
plt.xlabel('Day')
plt.ylabel('Reviews count')
#plt.show()

# Explorar cómo funciona TextBlob para polaridad:
# Con AFINN:
reviews['polarity'] = reviews['Reviews_Simp'].map(lambda text: TextBlob(text).sentiment.polarity)
reviews['review_len'] = reviews['Reviews_Simp'].astype(str).apply(len)
reviews['word_count'] = reviews['Reviews_Simp'].apply(lambda x: len(str(x).split()))
print(reviews.head())

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
#fig.show()

#Review lenght

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

#fig.show()
#Word count

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

#fig.show()

#Filtering data
positive_review = reviews[reviews["Sentiment"]=='Positive'].dropna()
neutral_review = reviews[reviews["Sentiment"]=='Neutral'].dropna()
negative_review = reviews[reviews["Sentiment"]=='Negative'].dropna()

## custom function for ngram generation ##
def generate_ngrams(text, n_gram = 1):
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
fig = subplots.make_subplots(rows=3, cols=1, vertical_spacing = 0.04,
                          subplot_titles=["Frequent words of positive reviews", "Frequent words of neutral reviews",
                                          "Frequent words of negative reviews"])
fig.add_trace(trace0, 1, 1)
fig.add_trace(trace1, 2, 1)
fig.add_trace(trace2, 3, 1)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
#iplot(fig, filename='word-plots')


# BIGRAMA
## Get the bar chart from positive reviews ##
freq_dict = defaultdict(int)
for sent in positive_review["Reviews_Simp"]:
    for word in generate_ngrams(sent, 2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(20), 'blue')


## Get the bar chart from neutral reviews ##
freq_dict = defaultdict(int)
for sent in neutral_review["Reviews_Simp"]:
    for word in generate_ngrams(sent, 2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(20), 'purple')


## Get the bar chart from negative reviews ##
freq_dict = defaultdict(int)
for sent in negative_review["Reviews_Simp"]:
    for word in generate_ngrams(sent, 2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(20), 'yellow')

# Creating two subplots
fig = subplots.make_subplots(rows=3, cols=1, vertical_spacing = 0.04,
                          subplot_titles=["Frequent words of positive reviews", "Frequent words of neutral reviews",
                                          "Frequent words of negative reviews"])
fig.add_trace(trace0, 1, 1)
fig.add_trace(trace1, 2, 1)
fig.add_trace(trace2, 3, 1)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
#iplot(fig, filename='word-plots')

# N = 3

## Get the bar chart from positive reviews ##
freq_dict = defaultdict(int)
for sent in positive_review["Reviews_Simp"]:
    for word in generate_ngrams(sent, 3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(20), 'blue')


## Get the bar chart from neutral reviews ##
freq_dict = defaultdict(int)
for sent in neutral_review["Reviews_Simp"]:
    for word in generate_ngrams(sent, 3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(20), 'purple')


## Get the bar chart from negative reviews ##
freq_dict = defaultdict(int)
for sent in negative_review["Reviews_Simp"]:
    for word in generate_ngrams(sent, 3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(20), 'yellow')

# Creating two subplots
fig = subplots.make_subplots(rows=3, cols=1, vertical_spacing = 0.04,
                          subplot_titles=["Frequent words of positive reviews", "Frequent words of neutral reviews",
                                          "Frequent words of negative reviews"])
fig.add_trace(trace0, 1, 1)
fig.add_trace(trace1, 2, 1)
fig.add_trace(trace2, 3, 1)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
#iplot(fig, filename='word-plots')

#WORDCLOUD

text = positive_review["Reviews_Simp"]
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
#plt.show()

text = neutral_review["Reviews_Simp"]
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
#plt.show()


text = negative_review["Reviews_Simp"]
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
#plt.show()

#Extracting features from reviews

# calling the label encoder function
le = preprocessing.LabelEncoder()

# Encode labels in column 'sentiment'.
reviews['Sentiment'] = le.fit_transform(reviews['Sentiment'])

reviews['Sentiment'].unique()

print(reviews['Sentiment'].value_counts())

#Extracting 'reviews' for processing
review_features = reviews.copy()
review_features = review_features[['Reviews_Simp']].reset_index(drop=True)
review_features.head()

#tf-idf

tfidf_vectorizer = TfidfVectorizer(max_features = 5000, ngram_range = (2,2))
# TF-IDF feature matrix
X = tfidf_vectorizer.fit_transform(review_features['Reviews_Simp'])
print(X.shape)

#Getting the target variable(encoded)
y = reviews['Sentiment']
print(y.shape)

#Oversampling

print(f'Original dataset shape : {Counter(y)}')

smote = SMOTE(random_state = 42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f'Resampled dataset shape {Counter(y_resampled)}')

# TRAIN - TEST

## Splitting the dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state=0)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#creating the objects
logreg = LogisticRegression(random_state=0)
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()
svc = SVC()
nb = BernoulliNB()
rf = RandomForestClassifier()
cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2:'KNN', 3:'SVC', 4:'Naive Bayes', 5: 'Random Forest'}
cv_models = [logreg, dt, knn, svc, nb, rf]


# Definir los parámetros a buscar para cada clasificador
param_grid = [
    # Parámetros para Logistic Regression
    {
        'C': np.logspace(-4, 4, 50),
        'penalty': ['l1', 'l2'],
        'max_iter':[1000]
    },
    # Parámetros para Decision Tree
    {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    # Parámetros para KNN
    {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    # Parámetros para SVC
    {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    },
    # Parámetros para Naive Bayes
    {
        'alpha': [0.1, 0.5, 1.0]
    },
    # Parámetros para Random Forest
    {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
]

# Crear una lista de los clasificadores y sus correspondientes parámetros
cv_models = [logreg, dt, knn, svc, nb, rf]

# Realizar la búsqueda de hiperparámetros y evaluación para cada clasificador
"""for i, (model, params) in enumerate(zip(cv_models, param_grid)):
    clf = GridSearchCV(model, params, cv=5, verbose=0, n_jobs=-1)
    best_model = clf.fit(X_train, y_train)
    print("{} Best Parameters: {}".format(cv_dict[i], best_model.best_params_))
    print("{} Test Accuracy: {:.2f}".format(cv_dict[i], best_model.best_score_))
    print("{} Test Accuracy on Test Set: {:.2f}".format(cv_dict[i], best_model.score(X_test, y_test)))
    print()
print(best_model.best_estimator_)
print("The mean accuracy of the model is:", best_model.score(X_test,y_test))

logreg = LogisticRegression(C = 1526.4179671752304, random_state = 0)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes = ['Positive','Neutral','Negative'])
print("Classification Report:\n",classification_report(y_test, y_pred))"""
# Crear una lista de los clasificadores entrenados con los mejores parámetros
best_models = []
results = []

# Entrenar y evaluar cada modelo
for i, (model, params) in enumerate(zip(cv_models, param_grid)):
    # Realizar la búsqueda de hiperparámetros
    clf = GridSearchCV(model, params, cv=5, verbose=0, n_jobs=-1)
    best_model = clf.fit(X_train, y_train)
    best_models.append(best_model.best_estimator_)
    print("{} Best Parameters: {}".format(cv_dict[i], best_model.best_params_))
    print("{} Test Accuracy: {:.2f}".format(cv_dict[i], best_model.best_score_))

    # Evaluar el modelo en el conjunto de prueba
    y_pred = best_model.predict(X_test)
    accuracy = best_model.score(X_test, y_test)
    print("{} Test Accuracy on Test Set: {:.2f}".format(cv_dict[i], accuracy))

    # Calcular precision, recall y F-score
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    precision_micro, recall_micro, fscore_micro, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')

    # Almacenar los resultados en la lista
    results.append({
        'Model': cv_dict[i],
        'Precision (macro)': precision,
        'Recall (macro)': recall,
        'F-score (macro)': fscore,
        'Precision (micro)': precision_micro,
        'Recall (micro)': recall_micro,
        'F-score (micro)': fscore_micro
    })

    # Calcular la matriz de confusión y mostrar el informe de clasificación
    cm = metrics.confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=['Positive', 'Neutral', 'Negative'])
    print("Classification Report for {}: \n{}".format(cv_dict[i], classification_report(y_test, y_pred)))
    print()