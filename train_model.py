import getopt
import pickle
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report  # Import desired metrics
# Other miscellaneous libraries
from imblearn.over_sampling import SMOTE
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from scipy.stats import pearsonr
from collections import Counter
from nltk.stem.porter import PorterStemmer
import re
import string

from textblob import TextBlob

stemmer = PorterStemmer()

INPUT_FILE = ""
OUTPUT_FILE = ""
TEXTO = "True"
MODELO = ""

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
              'singapore', 'airlines', 'airline']

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
    #  "MEL": "Melbourne",
    #  "SIN": "Singapore",
    #  "HND": "Tokyo",
    #  " New York": "New York",
    #  "New York JFK": "New York"
}


def usage():
    # PRE: ---
    # POST: se imprime por pantalla la ayuda del script y salimos del programa
    print("Uso de train_model.py <optional-args>")
    print("OPCIONES PARA ENTRENAR DATOS DE SINGAPUR")
    print(f"-h, --help      show the usage")
    print(f"-i, --input     input file path of the data   DEFAULT: ./{INPUT_FILE}")
    print(f"-o, --output    output file path for the predictions     DEFAULT: {OUT_FILE}")
    print(f"-m, --model     Modelo que se utilizará para las predicciones")
    print(f"-t, --text      ¿Se usará el texto como feature?  DEFAULT: {TEXTO} ")
    print("      --> Opciones (2): True / False")

    # Salimos del programa
    exit(1)


def load_options(options):
    global INPUT_FILE, OUTPUT_FILE, TEXTO, MODELO

    for opt, arg in options:
        if opt in ('-i', '--input'):
            INPUT_FILE = str(arg)
        elif opt in ('-o', '--output'):
            OUTPUT_FILE = str(arg)
        elif opt in ('-t', '--text'):
            TEXTO = str(arg)
        elif opt in ('-m', '--modelo'):
            MODELO = str(arg)
        elif opt in ('-h', '--help'):
            usage()


def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


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


def g_att_num(dict, city):
    return dict.get(city, 0)


def g_att_str(dict, city):
    return dict.get(city, "Unknown")


def csv_to_dict(file):
    df = pd.read_csv(file)
    data_dict = dict(zip(df['City'], df[df.columns[1]]))
    return data_dict


def main():
    # Step 1: Load the Pre-trained Model from the .sav File
    model_filename = MODELO  # Replace with your model's filename
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    # Step 2: load test set
    test = pd.read_csv(INPUT_FILE)

    # Step 3: procesar los datos para asegurarnos de que están iguales al entreno.
    if TEXTO == "True":
        ## print shape of dataset with rows and columns and information
        print("The shape of the  data is (row, column):" + str(test.shape))
        print("The Information about the dataset:" + str(test.info()))

        # Checking for null values
        print('The null values in the dataset:')
        print(test.isnull().sum())

        num_duplicates = test.duplicated().sum()  # identify duplicates
        print('There are {} duplicate reviews present in the dataset'.format(num_duplicates))

        # JUNTAR TITLE Y REVIEW
        test['Rev'] = test['Reviews'] + test['Title']
        test = test.drop(['Reviews', 'Title'], axis=1)
        print(test.head())

        # Simplificar texto según función clean_review
        test['Reviews_Simp'] = test["Rev"].apply(lambda x: clean_review(x))
        print(test.head())

        # Quitar stop words
        test['Reviews_Simp'] = test['Reviews_Simp'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
        print(test.head())

        # Contador de palabras
        cnt = Counter()
        for text in test["Reviews_Simp"].values:
            for word in text.split():
                cnt[word] += 1

        print(cnt.most_common(10))

        # Stem words: LOS RESULTADOS EMPEORAN

        # reviews["Reviews_Simp"] = reviews["Reviews_Simp"].apply(lambda text: stem_words(text))
        # print(reviews.head())

        # PROCESAMIENTO DE EMOTICONOS -> CONVERTIR A TEXTO
        # https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing

        # DATOS EXTRAÍDOS:
        test['polarity'] = test['Reviews_Simp'].map(lambda text: TextBlob(text).sentiment.polarity)
        test['review_len'] = test['Reviews_Simp'].astype(str).apply(len)
        test['word_count'] = test['Reviews_Simp'].apply(lambda x: len(str(x).split()))
        print(test.head())

        columns = ["Reviews_Simp"]
        test_features = test[columns].reset_index(drop=True)
        print(test_features.head())
        # tf-idf

        tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))
        # TF-IDF feature matrix
        tfidf_reviews = tfidf_vectorizer.fit_transform(test['Reviews_Simp'])

        # predicciones:
        predictions = model.predict(tfidf_reviews)

        results = test.copy()
        results['Sentiment_Pred'] = predictions

    results.to_csv(OUTPUT_FILE, index=False)


if __name__ == '__main__':
    print('ARGV   :', sys.argv[1:])
    try:
        options, reminder = getopt.getopt(sys.argv[1:], 'i:o:t:m:h',
                                          ['input=', 'output=', 'help', 'text=', 'model='])

    except getopt.GetoptError as err:
        print('ERROR:', err)
        sys.exit(1)
    print('OPTIONS:  ', options)
    load_options(options)
    main()
