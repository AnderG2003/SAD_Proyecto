import getopt
import os
import pickle
import re
import string
import sys
# Ignore warnings
import warnings

# NLTK libraries
from nltk.stem.porter import PorterStemmer
# Visualization libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
# Metrics libraries
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support, make_scorer, \
    f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
# Machine Learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from textblob import TextBlob

adasyn = ADASYN()

warnings.filterwarnings('ignore')

# Other miscellaneous libraries
from imblearn.over_sampling import SMOTE
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from scipy.stats import pearsonr
from collections import Counter

stemmer = PorterStemmer()

INPUT_FILE = "airlines_reviewsSingapore.csv"
OUT_FILE = "Modelos"
TEXTO = "False"
OVERSAMPLING = 'smote'

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


############################## FUNCIONES TECLADO ##############################################

def usage():
    # PRE: ---
    # POST: se imprime por pantalla la ayuda del script y salimos del programa
    print("Uso de sentiAnal.py <optional-args>")
    print("OPCIONES PARA ANÁLISIS DE SENTIMIENTOS DE AEROLÍNEA SINGAPUR")
    print(f"-h, --help      show the usage")
    print(f"-i, --input     input file path of the data   DEFAULT: ./{INPUT_FILE}")
    print(f"-o, --output    output file path for the weights            DEFAULT: ./{OUT_FILE}")
    print(f"-t, --text      ¿Se usará el texto como feature?  DEFAULT: ./{TEXTO} || True o False")
    print(f"-s, --sampling   Método para oversampling  DEFAULT: ./{OVERSAMPLING} || smote o adasyn")
    print("")

    # Salimos del programa
    exit(1)


def load_options(options):
    global INPUT_FILE, OUT_FILE, TEXTO, OVERSAMPLING

    for opt, arg in options:
        if opt in ('-i', '--input'):
            INPUT_FILE = str(arg)
        elif opt in ('-o', '--output'):
            OUT_FILE = str(arg)
        elif opt in ('-s', '--sampling'):
            OVERSAMPLING = str(arg)
        elif opt in ('-t', '--text'):
            TEXTO = str(arg)
        elif opt in ('-h', '--help'):
            usage()


########################## FUNCIONES #################################


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


def csv_to_dict(file):
    df = pd.read_csv(file)
    data_dict = dict(zip(df['City'], df[df.columns[1]]))
    return data_dict


def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

############### MAIN #################

def main():
    reviews = pd.read_csv(INPUT_FILE)
    print("Singapore Airline Dataset:")
    print(reviews.head(5))

    ## print shape of dataset with rows and columns and information
    print("The shape of the  data is (row, column):" + str(reviews.shape))
    print("The Information about the dataset:" + str(reviews.info()))

    # Lista de columnas a eliminar
    columns_to_drop = ['Seat Comfort', 'Staff Service', 'Food & Beverages', 'Inflight Entertainment', 'Value For Money',
                       'Airline', 'Recommended']
    print("Para SENTIMENT ANALYSIS no podrán utilizarse las columnas: ", columns_to_drop)
    # Comprobar si las columnas a eliminar existen en el DataFrame
    existing_columns = [col for col in columns_to_drop if col in reviews.columns]
    # Eliminar solo las columnas que existen
    if existing_columns:
        reviews.drop(existing_columns, axis=1, inplace=True)
    # Mostrar el DataFrame después de eliminar las columnas
    print("The shape of the  data is (row, column):" + str(reviews.shape))
    print("The Information about the dataset:" + str(reviews.info()))

    # Transformamos la columna Overall rating en Sentiment
    reviews["Sentiment"] = reviews.apply(sent, axis=1)
    reviews.drop("Overall Rating", axis=1, inplace=True)

    # Dividimos la ruta en origen y destino, corregimos la faltas ortográficas.
    reviews["Origin"] = reviews["Route"].apply(get_origin)
    reviews["Destiny"] = reviews["Route"].apply(get_destiny)
    # Verificamos
    print(reviews["Origin"].value_counts())
    print(reviews["Destiny"].value_counts())
    print(reviews["Verified"].value_counts())
    print(reviews["Class"].value_counts())

    reviews["Origin"] = reviews["Origin"].replace(corrections_dict)
    reviews["Destiny"] = reviews["Destiny"].replace(corrections_dict)
    reviews["Scale"] = reviews["Route"].apply(get_scale)
    reviews["Scale_bool"] = reviews["Scale"].apply(bool_scale)

    print(reviews["Origin"].value_counts())
    print(reviews["Destiny"].value_counts())

    unique_cities = set(reviews["Origin"]).union(set(reviews["Destiny"]))

    csv_files = ["city_country.csv", "city_continent.csv", "city_latitude.csv", "city_longitude.csv"]
    csv_existence = {}

    # Comprobar si existen los archivos CSV
    for file in csv_files:
        csv_existence[file] = os.path.exists(file)  # True si existe, False si no

    if all(not exists for exists in csv_existence.values()):
        city_country_dict = {}
        city_cont_dict = {}
        city_lat_dict = {}
        city_long_dict = {}

        for city in unique_cities:
            location_info = read_country(city)
            if location_info:
                city_country_dict[city] = location_info.get("country")  # Guardar país
                city_cont_dict[city] = location_info.get("continent")  # Guardar continente
                city_lat_dict[city] = location_info.get("latitude")  # Guardar latitud
                city_long_dict[city] = location_info.get("longitude")  # Guardar longitud
            else:
                print(f"Información no encontrada para la ciudad: {city}")

        # Crear DataFrames para cada diccionario
        df_country = pd.DataFrame(list(city_country_dict.items()), columns=['City', 'Country'])
        df_continent = pd.DataFrame(list(city_cont_dict.items()), columns=['City', 'Continent'])
        df_latitude = pd.DataFrame(list(city_lat_dict.items()), columns=['City', 'Latitude'])
        df_longitude = pd.DataFrame(list(city_long_dict.items()), columns=['City', 'Longitude'])

        # Exportar a archivos CSV
        df_country.to_csv("city_country.csv", index=False)
        df_continent.to_csv("city_continent.csv", index=False)
        df_latitude.to_csv("city_latitude.csv", index=False)
        df_longitude.to_csv("city_longitude.csv", index=False)
    else:
        city_country_dict = csv_to_dict("city_country.csv")
        city_cont_dict = csv_to_dict("city_continent.csv")
        city_lat_dict = csv_to_dict("city_latitude.csv")
        city_long_dict = csv_to_dict("city_longitude.csv")

    # Agregar columnas al DataFrame usando apply y lambda
    reviews["Or_country"] = reviews["Origin"].apply(lambda city: g_att_str(city_country_dict, city))
    reviews["Dst_country"] = reviews["Destiny"].apply(lambda city: g_att_str(city_country_dict, city))

    reviews["Or_continent"] = reviews["Origin"].apply(lambda city: g_att_str(city_cont_dict, city))
    reviews["Dst_continent"] = reviews["Destiny"].apply(lambda city: g_att_str(city_cont_dict, city))

    reviews["Or_lat"] = reviews["Origin"].apply(lambda city: g_att_num(city_lat_dict, city))
    reviews["Dst_lat"] = reviews["Destiny"].apply(lambda city: g_att_num(city_lat_dict, city))

    reviews["Or_long"] = reviews["Origin"].apply(lambda city: g_att_num(city_long_dict, city))
    reviews["Dst_long"] = reviews["Destiny"].apply(lambda city: g_att_num(city_long_dict, city))

    reviews["distance_km"] = reviews.apply(
        lambda row: geodesic(
            (row["Or_lat"], row["Or_long"]),
            (row["Dst_lat"], row["Dst_long"])
        ).km, axis=1
    )

    print(reviews.head())

    ## print shape of dataset with rows and columns and information
    print("The shape of the  data is (row, column):" + str(reviews.shape))
    print("The Information about the dataset:" + str(reviews.info()))

    # Checking for null values
    print('The null values in the dataset:')
    print(reviews.isnull().sum())

    num_duplicates = reviews.duplicated().sum()  # identify duplicates
    print('There are {} duplicate reviews present in the dataset'.format(num_duplicates))

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

    ################### PROCESAR TEXTO #######################

    if TEXTO == "True":
        # Simplificar texto según función clean_review
        reviews['Reviews_Simp'] = reviews["Rev"].apply(lambda x: clean_review(x))
        print(reviews.head())

        # Quitar stop words
        reviews['Reviews_Simp'] = reviews['Reviews_Simp'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
        print(reviews.head())

        # Contador de palabras
        cnt = Counter()
        for text in reviews["Reviews_Simp"].values:
            for word in text.split():
                cnt[word] += 1

        print(cnt.most_common(10))

        # Stem words: LOS RESULTADOS EMPEORAN

        #reviews["Reviews_Simp"] = reviews["Reviews_Simp"].apply(lambda text: stem_words(text))
        #print(reviews.head())

        #PROCESAMIENTO DE EMOTICONOS -> CONVERTIR A TEXTO
        #https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing

        # DATOS EXTRAÍDOS:
        reviews['polarity'] = reviews['Reviews_Simp'].map(lambda text: TextBlob(text).sentiment.polarity)
        reviews['review_len'] = reviews['Reviews_Simp'].astype(str).apply(len)
        reviews['word_count'] = reviews['Reviews_Simp'].apply(lambda x: len(str(x).split()))
        print(reviews.head())

    #################### FEATURES ##############################3

    # calling the label encoder function
    le = preprocessing.LabelEncoder()

    # Encode labels in column 'sentiment'.
    reviews['Sentiment'] = le.fit_transform(reviews['Sentiment'])

    reviews['Sentiment'].unique()

    print(reviews['Sentiment'].value_counts())
    # Positive -> 2
    # Neutral -> 1
    # Negative -> 0
    review_features = reviews.copy()
    ### TEXTO SÍ:
    # Extracting 'reviews' for processing
    if TEXTO == "True":
        columns = ["Reviews_Simp"]
        review_features = review_features[columns].reset_index(drop=True)
        print(review_features.head())

        # tf-idf

        tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))
        # TF-IDF feature matrix
        tfidf_reviews = tfidf_vectorizer.fit_transform(review_features['Reviews_Simp'])
        X = tfidf_reviews
        print(X.shape)
    else:
        columns = ["Origin", "Destiny", "Scale", "Verified",
                   "Type of Traveller", "Class", "Scale_bool", "year", "month", "day", "Or_country", "Dst_country",
                   "Or_lat", "Dst_lat", "Dst_long", "Or_long", "distance_km"]
        # columns = ["Reviews_Simp", "Origin", "Destiny", "Scale", "polarity", "review_len", "word_count", "Verified",
        #           "Type of Traveller", "Class", "Scale_bool"]
        review_features = review_features[columns].reset_index(drop=True)
        review_features["Verified"] = reviews["Verified"].astype(int)
        review_features["Scale_bool"] = reviews["Scale_bool"].astype(int)
        review_features["Class"] = le.fit_transform(review_features["Class"])
        review_features["Type of Traveller"] = le.fit_transform(review_features["Type of Traveller"])
        review_features["Origin"] = le.fit_transform(review_features["Origin"])
        review_features["Destiny"] = le.fit_transform(review_features["Destiny"])
        review_features["Scale"] = le.fit_transform(review_features["Scale"])
        review_features["Or_country"] = le.fit_transform(review_features["Or_country"])
        review_features["Dst_country"] = le.fit_transform(review_features["Dst_country"])
        review_features["year"] = review_features["year"].astype("int")
        review_features["month"] = review_features["month"].astype("int")
        review_features["day"] = review_features["day"].astype("int")
        # review_features["Or_continent"] = le.fit_transform(review_features["Or_continent"])
        # review_features["Dst_continent"] = le.fit_transform(review_features["Or_continent"])
        print(review_features.head())

        # Numerical:
        #numerical_ft = review_features[["Origin", "Destiny", "Scale", "Verified",
        #  "Type of Traveller", "Class", "Scale_bool"]]

        #numerical_ft = review_features[["Origin", "Destiny", "Scale", "Verified",
                                     #   "Type of Traveller", "Class", "Scale_bool", "Or_country", "Dst_country", "year",
                                      #  "month", "day", "Or_lat", "Dst_lat", "Or_long", "Dst_long", "distance_km"]]

        # LAS MEJORES PARA SVC:
        #numerical_ft = review_features[["Origin", "Destiny", "Scale", "Verified",
         #                           "Type of Traveller", "Scale_bool", "Or_country", "Dst_country", "year", "Class", "day",
         #                        "distance_km", "month"]]

        #LAS MEJORES PARA KNN:
        numerical_ft = review_features[["Origin", "Destiny", "Scale", "Verified","Type of Traveller", "Class", "Scale_bool", "Or_country", "Dst_country", "year",
    "Or_lat", "Dst_lat", "Or_long", "Dst_long", "month", "day"]]
        scaler = StandardScaler()
        num_sc = scaler.fit_transform(numerical_ft)
        # importancia de features:
        correlations = {}
        for column in numerical_ft.columns:
            # Asegúrate de que ambas columnas son numéricas
            if numerical_ft[column].dtype in ['int64', 'float64']:
                correlation, _ = pearsonr(numerical_ft[column], reviews['Sentiment'])
                correlations[column] = correlation
            else:
                print(f"La columna '{column}' no es numérica y no se puede calcular la correlación.")

        # Ordenar las características por valor de correlación
        # important_features = sorted(correlations, key=correlations.get, reverse=True)

        # print("Características ordenadas por correlación:", important_features)
        # Crear una lista de tuplas con la característica y su correlación
        feature_correlations = [(col, correlations[col]) for col in
                                sorted(correlations, key=correlations.get, reverse=True)]

        # Mostrar la lista de características y sus correlaciones
        print("Características ordenadas por correlación con su valor de correlación:")
        for feature, correlation in feature_correlations:
            print(f"{feature}: {correlation:.3f}")

        X = num_sc
        print(X.shape)

    y = reviews['Sentiment']
    print(y.shape)

    # Oversampling o undersampling? En este caso OVER

    print(f'Original dataset shape : {Counter(y)}')
    if OVERSAMPLING == 'smote':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    else:
        adasyn = ADASYN()
        X_resampled, y_resampled = adasyn.fit_resample(X, y)

    print(f'Resampled dataset shape {Counter(y_resampled)}')

    # TRAIN - TEST

    ## Splitting the dataset into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

    ###################### MODELADO #######################
    # creating the objects
    logreg = LogisticRegression(random_state=0)
    dt = DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    svc = SVC()
    nb = BernoulliNB()
    rf = RandomForestClassifier()
    cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'KNN', 3: 'SVC', 4: 'Naive Bayes', 5: 'Random Forest'}
    cv_models = [logreg, dt, knn, svc, nb, rf]

    # Definir los parámetros a buscar para cada clasificador
    param_grid = [
        # Parámetros para Logistic Regression
        {
            'C': np.logspace(-4, 4, 50),
            'penalty': ['l1', 'l2'],
            'max_iter': [1000]
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

    # Crear una lista de los clasificadores entrenados con los mejores parámetros
    best_models = []
    results = []
    cv_results_dfs = []

    # Entrenar y evaluar cada modelo
    for i, (model, params) in enumerate(zip(cv_models, param_grid)):

        # GridSearch -> cv = 5 (estándar), cv = 10 (para conjuntos pequeños, más preciso)
        # No hay mejoras significativas entre cv=5 y cv=10, lo dejamos en 5.
        # cv = cross-validation. Se divide train en 5 trozos; se utilizan 4/5 para train, 1/5 para dev. Se dan
        # 5 vueltas. De esta manera se evita el overfitting.

        clf = GridSearchCV(model, params, cv=5, scoring=make_scorer(f1_score, average='weighted'), verbose=0, n_jobs=-1)
        best_model = clf.fit(X_train, y_train)
        best_models.append(best_model.best_estimator_)

        print("{} Best Parameters: {}".format(cv_dict[i], best_model.best_params_))
        print("{} Test Accuracy: {:.2f}".format(cv_dict[i], best_model.best_score_))

        # Evaluar el modelo en el conjunto de prueba
        y_pred = best_model.predict(X_test)
        accuracy = best_model.score(X_test, y_test)
        print("{} Test Accuracy on Test Set: {:.2f}".format(cv_dict[i], accuracy))

        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        results.append({
            'Model_Name': cv_dict[i],
            'Best_Score': fscore,
            'Test_Accuracy': accuracy,
            'Best_Params': best_model.best_params_
        })

        cv_results = pd.DataFrame(clf.cv_results_)
        for j in range(len(cv_results)):
            y_pred = best_model.predict(X_test)  # Pr
            precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
            precision_micro, recall_micro, fscore_micro, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                             average='micro')

            # Crear DataFrame para resultados específicos
            results_df = pd.DataFrame({
                'Model': cv_dict[i],  # Nombre del modelo
                'Params': str(cv_results.loc[j, 'params']),  # Hiperparámetros
                'Class': ['Positive', 'Neutral', 'Negative'],  # Clases
                'Precision': precision,
                'Recall': recall,
                'F-score': fscore,
                'micro-prec': precision_micro,
                'micro-recall': recall_micro,
                'micro-fscore': fscore_micro
            })
            cv_results = pd.DataFrame(clf.cv_results_)
            # Agregar DataFrame a la lista de resultados
            cv_results_dfs.append(results_df)

        # Calcular la matriz de confusión y mostrar el informe de clasificación
        cm = metrics.confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, classes=['Positive', 'Neutral', 'Negative'])
        print("Classification Report for {}: \n{}".format(cv_dict[i], classification_report(y_test, y_pred)))
        print()

    final_results_df = pd.concat(cv_results_dfs, ignore_index=True)

    # Guardar en un archivo CSV
    final_results_df.to_csv('final_results.csv', index=False)

    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv("metricas_act.csv")
    df_metrics = df_metrics.sort_values(by='Best_Score', ascending=False)

    if TEXTO == "True":
        for idx, model_info in df_metrics.head(2).iterrows():
            model_name = model_info['Model_Name']
            best_model = best_models[idx]

            # Guardar el modelo con pickle
            with open(f"{OUT_FILE}/modelo_{model_name}.sav", 'wb') as f:
                pickle.dump(best_model, f)
                print(f"Modelo {model_name} guardado como modelo_{model_name}.sav")
    else:
        for idx, model_info in df_metrics.head(1).iterrows():
            model_name = model_info['Model_Name']
            best_model = best_models[idx]

            # Guardar el modelo con pickle
            with open(f"{OUT_FILE}/modelo_{model_name}.sav", 'wb') as f:
                pickle.dump(best_model, f)
                print(f"Modelo {model_name} guardado como modelo_{model_name}.sav")



    reviews.to_csv("sing_prepared.csv", index=False)


if __name__ == '__main__':
    print('ARGV   :', sys.argv[1:])
    try:
        options, reminder = getopt.getopt(sys.argv[1:], 'i:o:t:s:h',
                                          ['input=', 'output=', 'help', 'text=', 'oversampling='])

    except getopt.GetoptError as err:
        print('ERROR:', err)
        sys.exit(1)
    print('OPTIONS:  ', options)
    load_options(options)
    main()
