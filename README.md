# SAD_Proyecto

Bidane León

Patricia Ortega

Asier Larrazabal 

Alessandra Taipe

Eneko Uresandi

Ander Gorocica

## Llamada Sentimen Analysis:
#### Opciones:
```
(-t -> ¿Se utiliza el texto para entrenar? True / False)
(-s -> TÉCNICA DE OVERSAMPLING: smote / adasyn / smoteenn / smotetomek)
(-o -> Carpeta donde se guardan los modelos, debe existir)
(-i -> archivo a analizar)
(-m -> la métrica según la cual se juzgará el mejor modelo: fscore / precision /accuracy / recall)
(-c -> clase cuyos resultados queremos optimizar: none / pos / neg)
```
#### Ejemplo:
```console
foo@bar:~$  python sentiMenu.py -i sing_entero.csv -o ModelosTexto -s smote -t True -m fscore 
```
#### Para texto:
```console
foo@bar:~$  python sentiMenu.py -i sing_entero.csv -o ModelosTexto -s smote -t True -m fscore 
```

#### Sin texto:
```console
foo@bar:~$  python sentiMenu.py -i sing_entero.csv -o ModelosTexto -s smote -t False -m fscore 
```
#### Para texto optimizando resultados positivos:
```console
foo@bar:~$  python sentiMenu.py -i sing_entero.csv -o ModelosTexto -s smote -t True -c pos
```


## Sentiment Analysis
To-do:
- [x] Preproceso
  - [x] Lematizar, normalizar, simplificar, quitar palabras habituales (?)
  - [x] Separar ruta en origen - destino - escalas
  - [x] Unir comentario y title
  - [x] Latitud longitud ?
  - [x] Oversampling
  - [x] Train - dev - test
  - [x] Overall Rating -> Sentiment
  - [ ] Seleccionar las features relevantes (Filtrado, wrapped, PCA...)
    - [x] Correlación
    - [ ] Automatizar la selección
- [x] Algoritmos:
  - [x] Logistic Regression
  - [x] KNN
  - [x] Naive-Bayes
  - [x] SVC
  - [x] Random Forest
  - [x] Decision Trees
- [x] Hacer el código llamable
  - [x] -i input
  - [x] -o output
  - [x] etc
- [x] Guardar los mejores modelos con pickle
- [x] Separar los dos clasificadores

## Resultados (21-04-2024)
![Logistic Regression](/images/lr.png)
![Decision Tree](/images/dt.png)
![Random Forest](/images/rf.png)
![KNN](/images/knn.png)
![SVC](/images/svc.png)
![Naive Bayes](/images/nb.png)

### Referencias:
[Sentiment Analysis Amazon](https://www.kaggle.com/code/soniaahlawat/sentiment-analysis-amazon-review#Review-Text-Word-Count-Distribution)
