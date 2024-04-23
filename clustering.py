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

    nombreTop = ["","","","","","","","","",""]
    nombreDatos = sys.argv[1]
    numTopicos = int(sys.argv[2])
    sentimiento = sys.argv[3]   ## CONSULTAR CON SENTIMENTANALYSIS (0 = negativo, 1 = neutral, 2 = positivo)

    if numTopicos > 10:
        print("No es posible realizar una ejecución con tantos tópicos.\nRepita la prueba con un número menor por favor.\n")
        exit(0)
    elif len(nombreTop) != numTopicos:
        print("Describe cada tópico lo que es")
        exit(0)

    # Cargar los datos del .csv
    ml_dataset = pd.read_csv(nombreDatos, header=0)
    ml_dataset = ml_dataset[~ml_dataset["Reviews"].isnull()]

    # FUTUROS FILTRADOS DE DATOS AQUÍ
    '''
    # Solo tener en cuenta las valoraciones con sentimiento concreto
    ml_dataset = ml_dataset[ml_dataset["NOMBRECOLUMNA"] == sentimiento]]  ## CONSULTAR CON SENTIMENTANALYSIS
    '''

    copia = ml_dataset
    
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
    documentosAux3 = []
    for docuAct in documentos:
        filtered_doc = []
        for token in docuAct:
            if token not in stopWords:
                filtered_doc.append(token)
        documentosAux3.append(filtered_doc)

    documentos = documentosAux3

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
    documentosAux4 = []
    for docuAct in documentos:
        filtered_doc = []
        for token in docuAct:
            if not token.isnumeric():
                filtered_doc.append(token)
        documentosAux4.append(filtered_doc)

    documentos = documentosAux4


    '''
    bigrama = Phrases(documentos, min_count=20)
    for ind in range(len(documentos)):
        for token in bigrama[documentos[ind]]:
            if '_' in token:
                documentos[ind].append(token)
    '''

    # Creamos un diccionario con los documentos
    diccionario = Dictionary(documentos)

    # Filtrar palabras que aparezcan menos de X veces, o más del 5% de los documentos
    diccionario.filter_extremes(no_below=20, no_above=0.05)

    # Parametros necesarios para crear el modelo
    for docuAct in documentos:
        documentosAux5.append(diccionario.doc2bow(doc))

    corpus = documentosAux5
    id2word = diccionario.id2token
    num_topics = numTopicos
    chunksize = 2000
    passes = 10
    iterations = 500
    #eval_every = 10

    # Creamos el modelo usando LDA
    modelo = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every= None,
        random_state = 1000
    )

    top_topics = modelo.top_topics(corpus)

    #Calcular coherencia respecto a cierto numero de clusters
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    # Mostrar tópicos
    
    print(len(top_topics))
    for i in range(0,int(numTopicos)):
        print("_____________________________________________")
        print(top_topics[i])


    print("Asumiendo que cada documento solo entra en un tópico, cantidades de documentos por tópico:")
    
    # Calcular número de documentos por tópico

    arrayTop = []
    i = 0
    lis = [0] * len(top_topics)
    
    while i != len(corpus):
        j = 0
        maxi = -1
        base = -1
        
        valores = modelo.get_document_topics(corpus[i], minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
        for val in valores:
            if val[1] > maxi:
                maxi = val[1]
                base = j
                        
            j = j + 1
        arrayTop.append(nombreTop[base])
        lis[base] = lis[base] + 1
        i = i + 1
        
    # Mostrarlo

    print(lis)

    copia["razón negativa"] = np.array(arrayTop).tolist()
    copia.to_csv("archivo.csv")
