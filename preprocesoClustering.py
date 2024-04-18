import re, string
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
from sklearn.neighbors import KNeighborsClassifier
import pickle
import csv
from sklearn.tree import DecisionTreeClassifier
from opencage.geocoder import OpenCageGeocode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':

    def coerce_to_unicode(x):
    if sys.version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x, 'utf-8')
        else:
            return unicode(x)
    else:
        return str(x)

    nombreImput = sys.argv[1]
    nombreOutput = sys.argv[2]

    # Cargar los datos
    ml_dataset = pd.read_csv(nombreImput)
    ml_dataset = ml_dataset[["Title","Name","Review Date","Airline","Verified","Reviews","Type of Traveller","Month Flown","Route","Class","Seat Comfort","Staff Service","Food & Beverages","Inflight Entertainment","Value For Money","Overall Rating","Recommended"]]
