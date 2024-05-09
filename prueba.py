import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
import matplotlib.pyplot as plt
# VARIABLES GLOBALES
SENTIMIENTO = 2 # SENTIMIENTO: 0= negativo, 1 = neutro, 2= positivo
TIPO_COHERENCIA = 'u_mass' # PUEDE SER TIPO_COHERENCIA 'u_mass' o 'c_v' 

# Cargar los datos desde el archivo CSV
data = pd.read_csv('sing_prepared.csv', header=0)

data = data[data['Sentiment'] == SENTIMIENTO]

# Preprocesamiento de tus datos, tokenización, eliminación de stopwords, etc.

# Crear un diccionario y un corpus
#text_data = data['Reviews_Simp'].tolist()  # Ajusta 'columna_texto' al nombre de tu columna
#dictionary = Dictionary(text_data)
#corpus = [dictionary.doc2bow(text) for text in text_data]

documentos = data['Reviews_Simp'].tolist()

# Nos aseguramos de que está todo en minusculas
for i in range(0, len(documentos)):
    documentos[i] = documentos[i].lower()

### PRUEBA 
documentos = [d.split() for d in documentos]

# Quitar palabras que son solo números
documentos = [[token for token in doc if not token.isnumeric()] for doc in documentos]

# Quitar palabras que pertenecen al stopWords

# Quitar palabras de un caracter
documentos = [[token for token in doc if len(token) > 1] for doc in documentos]

dictionary = Dictionary(documentos)

corpus = [dictionary.doc2bow(doc) for doc in documentos]
text_data = documentos

# Lista para almacenar las puntuaciones de coherencia
coherence_scores = []

# Entrenar varios modelos LDA con diferentes valores de K
for k in range(2, 22):  # Puedes ajustar el rango según tu preferencia
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=42)
    coherence_model = CoherenceModel(model=lda_model, texts=text_data, dictionary=dictionary, coherence=TIPO_COHERENCIA)
    coherence_score = coherence_model.get_coherence()
    coherence_scores.append(coherence_score)

# Graficar la puntuación de coherencia en función de K
plt.plot(range(2, 22), coherence_scores)
plt.xlabel("Número de Temas (K)")
plt.ylabel("Puntuación de Coherencia")
plt.title("Puntuación de Coherencia en función de K")
plt.xticks(range(2, 22))
plt.grid(True)
plt.show()
