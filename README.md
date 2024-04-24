# SAD_Proyecto

Bidane León

Patricia Ortega

Asier Larrazabal 

Alessandra Taipe

Eneko Uresandi

Ander Gorocica

##Llamada:
 > python sentiMenu.py -i INPUT_FILE -o OUT_DIR -s smote/adasyn -t True/False

(-t -> ¿Se utiliza el texto para entrenar?)
(-s -> TÉCNICA DE OVERSAMPLING)
(-o -> Carpeta donde se guardan los modelos, debe existir)
(-i -> archivo a analizar)

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
- [ ] Hacer el código llamable
  - [ ] -i input
  - [ ] -o output
  - [ ] etc
- [ ] Guardar los mejores modelos con pickle
- [ ] Separar los dos clasificadores

## Resultados (21-04-2024)
![Logistic Regression](/images/lr.png)
![Decision Tree](/images/dt.png)
![Random Forest](/images/rf.png)
![KNN](/images/knn.png)
![SVC](/images/svc.png)
![Naive Bayes](/images/nb.png)

### Referencias:
[Sentiment Analysis Amazon](https://www.kaggle.com/code/soniaahlawat/sentiment-analysis-amazon-review#Review-Text-Word-Count-Distribution)
