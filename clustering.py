import getopt
from sys import exit, argv, version_info
import os
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

import logging
from gensim.corpora import Dictionary
from gensim.models import LdaModel
#from nltk.tokenize import RegexpTokenizer
#from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.models import CoherenceModel


# VARIABLES GLOBALES
INPUT_PATH  =   "./datos/archivo.csv"           # Path de los archivos de entrada
OUTPUT_PATH =   "./modelos"                     # Path de los archivos de salida
TARGET_COL  =   "Sentiment"                     # Nombre de la columna a clasificar
SENTIMIENTO =   1                               # Sentimiento a filtrar (0 = Negativo, 1 = Neutro, 2 = Positivo)
#SAMPLING    =   "NO"                            # Metodo de muestreo del dataset (NO, UNDERSAMPLING, OVERSAMPLING)
ATRIBUTOS   =   ['Reviews_Simp']                # Atributos que seleccionamos del dataset (TODOS o lista)
#AEROLINEA   =   "Singapore Airlines"           # Aerolinea de la que miraremos los datos

###########################################################
#               CONFIGURACIONES                           #
###########################################################
def usage():
    print("Uso: python clustering.py <opciones>")
    print("Las posibles opciones son")
    print(f"-h, --help          muestar el uso")
    print(f"-i, --input         path de los archivos de entrada                 DEFAULT: ./{INPUT_PATH}")
    print(f"-o, --output        path de los archivos de salida                  DEFAULT: ./{OUTPUT_PATH}")
    print(f"-t, --target        nombre objetivo a predecir                      DEFAULT: {TARGET_COL}")
    print(f"-s, --sentiment     sentimiento a filtrar                           DEFAULT: {int(SENTIMIENTO)}")
    #print(f"-a, --airline       filtro de aerolinea                             DEFAULT: {AEROLINEA}")
    #print(f"-m                  estrategia de muestreo                          DEFAULT: {SAMPLING}")
    
    exit(1)

def cargar_opciones(options):
    global INPUT_PATH, OUTPUT_PATH, TARGET_COL, SENTIMIENTO#, SAMPLING

    for opt, arg in options:
        if opt in ('-h', '--help'):
            usage()
        elif opt in ('-i', '--input'):
            INPUT_PATH = str(arg)
        elif opt in ('-o', '--output'):
            OUTPUT_PATH = str(arg)    
        elif opt in ('-t', '--target'):
            TARGET_COL = str(arg)
        elif opt in ('-s','--sentimiento'):
            SENTIMIENTO = str(arg)
            int(SENTIMIENTO)

        #elif opt == '-m':
        #    SAMPLING = str(arg)
        #    if SAMPLING == '':
        #        print("ERROR: Debe especificar un valor para el muestreo.")
        #    elif SAMPLING not in ("OVERSAMPLING", "UNDERSAMPLING", "NO"):
        #        print(f"ERROR: El valor de muestreo {SAMPLING} no es valido. Introduzca OVERSAMPLING', 'UNDERSAMPLING' o 'NO'")
        #elif opt in ('-a','--airline'):
        #    AEROLINEA = str(arg)

def mostrar_opciones():
    print("clustering.py configuraciones:")
    print(f"-i                   path de los archivos de entrada:       {INPUT_PATH}")
    print(f"-o                   path de los archivos de salida:        {OUTPUT_PATH}")
    print(f"-t                   nombre objetivo a predecir:            {TARGET_COL}")
    print(f"-s                   sentimiento a filtrar:                 {SENTIMIENTO}")
    #print(f"-a                   aerolinea a filtrar:                   {AEROLINEA}")
    #print(f"-m                   estrategia de muestreo:                {SAMPLING}")
    print(f"                     atributos seleccionados:               {ATRIBUTOS}")


###########################################################
#               METODOS                                   #
###########################################################
def coerce_to_unicode(x):
    if version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x, 'utf-8')
        else:
            return unicode(x)
    return str(x)

def crear_path_salida():
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    for archivo in os.listdir(OUTPUT_PATH):
        path = os.path.join(OUTPUT_PATH, archivo)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print('No se ha podido eliminar algun archivo de la carpeta de salida')

'''
def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents, og):
    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        print(''.join([' ' +feature_names[i] + ' ' + str(round(topic[i], 5)) #y esto también
                for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        docProbArray=np.argsort(W[:,topic_idx])
    #    print(docProbArray)
   #     howMany=len(docProbArray);
    #    print("How Many");
    #    print(howMany);
        for doc_index in top_doc_indices:
            print("* " + og[doc_index])
'''         

# LLAMADA POR TERMINAL: python clustering.py nombreDeCSV numTopics
def main():
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

    print("Ejecución principal iniciando")
    crear_path_salida()
    #Cargar datos de entrada 
    ml_dataset = pd.read_csv(INPUT_PATH, header=0)
    '''
    if ATRIBUTOS == "TODOS":
        atr = ml_dataset.columns
    else:
        atr = ATRIBUTOS
    '''
    #ml_dataset = ml_dataset[atr]

    # Filtros seleccionados
    #ml_dataset = ml_dataset[ml_dataset['airline'] == AEROLINEA]

    #ml_dataset = ml_dataset[ml_dataset['Sentiment'] == SENTIMIENTO]
    #ml_dataset = ml_dataset[~ml_dataset['Reviews_Simp'].isnull()]
    ml_dataset = ml_dataset[ml_dataset[TARGET_COL] == int(SENTIMIENTO)]
    ml_dataset = ml_dataset[~ml_dataset['Reviews_Simp'].isnull()]
    copia = ml_dataset
    print("Cuantos Documentos van a analizarse: ", len(ml_dataset))
    


# Trabajamos con los textos de las reviews
    #documentos = ml_dataset["Reviews_Simp"].toList()
    documentos = ml_dataset['Reviews_Simp'].tolist()

    # Nos aseguramos de que está todo en minusculas
    for i in range(0, len(documentos)):
        documentos[i] = documentos[i].lower()

    ### PRUEBA 
    documentos = [d.split() for d in documentos]

    # Quitar palabras que son solo números
    documentos = [[token for token in doc if not token.isnumeric()] for doc in documentos]
    
    # Quitar palabras que pertenecen al stopWords
    documentos = [[token for token in doc if not token in stop_words] for doc in documentos]

    # Quitar palabras de un caracter
    documentos = [[token for token in doc if len(token) > 1] for doc in documentos]


    bigrama = Phrases(documentos, min_count=20)
    for ind in range(len(documentos)):
        for token in bigrama[documentos[ind]]:
            if '_' in token:
                documentos[ind].append(token)

    # Crear diccionario
    dictionary = Dictionary(documentos)
    
    # Filtrar palabras que aparezcan menos de X veces, o más del 5% de los documentos
    # Sentimiento: 0 -> 244 docs
    #              1 -> 93 docs
    #              2 -> 462 docs
    if SENTIMIENTO==0:
        dictionary.filter_extremes(no_below=5, no_above=0.2)
    elif SENTIMIENTO==1:
        dictionary.filter_extremes(no_below=5, no_above=0.2)
    elif SENTIMIENTO==2: 
        dictionary.filter_extremes(no_below=20, no_above=0.05)
    
    corpus = [dictionary.doc2bow(doc) for doc in documentos]

    temp = dictionary[0] 
    id2word = dictionary.id2token

    ## PARAMETROS LDA ##
    chunksize = 1000
    passes = 10
    iterations = 500
    eval_every = None
    alpha = 'auto' 
    eta = 'auto' 
    random_state = 1000

    if len(copia) < 100: 
        min = 3
        max = 7

    elif len(copia) < 300:
        min = 6
        max = 10
    
    else:
        min = 8
        max = 12


#############################################################
#                                                           #
#############################################################
    for i in range(min, max):
        num_topics = i

        modelo = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha=alpha,
        eta=eta,
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every= eval_every,
        random_state = random_state
        )

        conf_modelo = {'n_topics': num_topics,
                        'chunksize': chunksize,
                        'passes':passes,
                        'iterations':iterations,
                        'alpha':alpha
                        }
        
        top_topics = modelo.top_topics(corpus)

        media_coherencia = sum([t[1] for t in top_topics]) / num_topics
        print('La coherencia media para ', num_topics,' es:'' %.4f.' % media_coherencia)

    # Mostrar tópicos
    
    print(len(top_topics))
    for i in range(0,int(num_topics)):
        print("_____________________________________________")
        print(top_topics[i])


  #  print("Asumiendo que cada documento solo entra en un tópico, cantidades de documentos por tópico:")
    
    # Calcular número de documentos por tópico
'''
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

'''



if __name__ == '__main__':
    try:
        # options: registra los argumentos del usuario
        # remainder: registra los campos adicionales introducidos -> entrenar_knn.py esto_es_remainder
        options, remainder = getopt.getopt(argv[1:], 'h:i:o:t:s:m', ['help', 'input', 'output', 'target', 'sentimiento='])
        
    except getopt.GetoptError as err:
        # Error al parsear las opciones del comando
        print("ERROR: ", err)
        exit(1)

    print(options)

    cargar_opciones(options)

    mostrar_opciones()

    main()


'''
if __name__ == '__main__':

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

    #nombreTop = ["","","","","","","","","",""]
    nombreDatos = sys.argv[1]
    numTopicos = int(sys.argv[2])
    sentimiento = int(sys.argv[3])   ## CONSULTAR CON SENTIMENTANALYSIS (0 = negativo, 1 = neutral, 2 = positivo)

    nombreTop = [""] * numTopicos
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@: ", nombreTop)

    print("Los parametros que se han indicado son:\n numTopicos: ", numTopicos, "\n sentimiento: ", sentimiento)

    if numTopicos > 10:
        print("No es posible realizar una ejecución con tantos tópicos.\nRepita la prueba con un número menor por favor.\n")
        exit(0)
    elif len(nombreTop) != numTopicos:
        print("Describe cada tópico lo que es")
        exit(0)

    # Cargar los datos del .csv
    ml_dataset = pd.read_csv(nombreDatos, header=0)
    #ml_dataset = ml_dataset[~ml_dataset['Reviews_Simp'].isnull()]

    # FUTUROS FILTRADOS DE DATOS AQUÍ
    
    # Solo tener en cuenta las valoraciones con sentimiento concreto
    #ml_dataset = ml_dataset[ml_dataset["NOMBRECOLUMNA"] == sentimiento]]  ## CONSULTAR CON SENTIMENTANALYSIS
    
    # Filtrar las reseñas según el sentimiento

    ml_dataset = ml_dataset[ml_dataset['Sentiment'] == sentimiento]
    copia = ml_dataset
    print("Cuantos Documentos van a analizarse: ", len(ml_dataset))
    
    # Trabajamos con los textos de las reviews
    #documentos = ml_dataset["Reviews_Simp"].toList()
    documentos = ml_dataset['Reviews_Simp'].tolist()

    # Nos aseguramos de que está todo en minusculas
    for i in range(0, len(documentos)):
        documentos[i] = documentos[i].lower()

    ### PRUEBA 
    documentos = [d.split() for d in documentos]

    # Quitar palabras que son solo números
    documentos = [[token for token in doc if not token.isnumeric()] for doc in documentos]
    
    # Quitar palabras que pertenecen al stopWords
    documentos = [[token for token in doc if not token in stop_words] for doc in documentos]

    # Quitar palabras de un caracter
    documentos = [[token for token in doc if len(token) > 1] for doc in documentos]

    # Crear diccionario
    dictionary = Dictionary(documentos)
    
    # Filtrar palabras que aparezcan menos de X veces, o más del 5% de los documentos
    # Sentimiento: 0 -> 244 docs
    #              1 -> 93 docs
    #              2 -> 462 docs
    if sentimiento==0:
        dictionary.filter_extremes(no_below=5, no_above=0.2)
    elif sentimiento==1:
        dictionary.filter_extremes(no_below=5, no_above=0.2)
    elif sentimiento==2: 
        dictionary.filter_extremes(no_below=20, no_above=0.05)
    
    corpus = [dictionary.doc2bow(doc) for doc in documentos]

    temp = dictionary[0] 
    id2word = dictionary.id2token

    num_topics = numTopicos
    chunksize = 1000
    passes = 10
    iterations = 500
 

    #######################################################################

    bigrama = Phrases(documentos, min_count=20)
    for ind in range(len(documentos)):
        for token in bigrama[documentos[ind]]:
            if '_' in token:
                documentos[ind].append(token)
    
    #############################################################################3

    
    # Creamos el modelo usando LDA
    modelo = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha=1/num_topics,
        eta=1/num_topics,
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
    copia.to_csv("output.csv")
'''
