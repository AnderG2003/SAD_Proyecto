import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
import csv
from opencage.geocoder import OpenCageGeocode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

# LLAMADA POR TERMINAL: python clustering.py nombreDeCSV numTopics
if __name__ == '__main__':

    nombreDatos = sys.argv[1]
    numTopicos = sys.argv[2]
    sentimiento = sys.argv[3]   ## CONSULTAR CON SENTIMENTANALYSIS (0 = negativo, 1 = neutral, 2 = positivo)

    if numTopicos > 10:
        print("No es posible realizar una ejecución con tantos tópicos.\nRepita la prueba con un número menor por favor.\n")
        exit(0)

    # Cargar los datos del .csv
    ml_dataset = pd.read_csv(nombreDatos, header=0)
    ml_dataset = ml_dataset[~ml_dataset["Reviews"].isnull()]

    # FUTUROS FILTRADOS DE DATOS AQUÍ
    '''
    # Solo tener en cuenta las valoraciones con sentimiento concreto
    ml_dataset = ml_dataset[ml_dataset["NOMBRECOLUMNA"] == sentimiento]]  ## CONSULTAR CON SENTIMENTANALYSIS
    '''

    # Trabajamos con los textos de las reviews
    documentos = ml_dataset["Reviews"].toList()

    # Nos aseguramos de que está todo en minusculas
    for i in range(0, len(documentos)):
        documentos[i] = documentos[i].lower()

    # Por cada review, creamos una lista con las palabras de la review
    documentosAux = []
    for docuAct in documentos:
        documentosAux.append(d.split())

    documentos = documentosAux
    
    # De cada lista quitamos las palabras que hemos considerado stopwords
    new_docs = []
    for doc in docs:
        filtered_doc = []
        for token in doc:
            if token not in stopWords:
                filtered_doc.append(token)
        new_docs.append(filtered_doc)

    docs = new_docs
    # De cada lista quitamos las palabras de longitud 1
    documentosAux2 = []
    for docuAct in documentos:
        token_filtrados = []
        for token in docuAct:
            if len(token) > 1:
                token_filtrados.append(token)
        documentosAux2.append(token_filtrados)

    documentos = documentosAux2

    # De cada lista quitamos las palabras que son solo números


    