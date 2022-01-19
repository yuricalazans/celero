# -*- coding: utf-8 -*-

import sys

if len(sys.argv) < 5:
  exit()

path_treino = sys.argv[2]
path_modelo = sys.argv[4]

## etapa 1 - carga de dados

import pandas as pd
import numpy as np
import os

def ler_arquivo_texto(path_arquivo):
  with open(path_arquivo, 'r') as f:
    try:
      aux += f.read()
    except NameError:
      aux = f.read()
  return aux

data = []
data_labels = []

for subdir in ['pos','neg']:
  path_aux = path_treino + subdir
  os.chdir(path_aux)

  for file in os.listdir():
    path_arquivo = f"{path_aux}/{file}"
    texto = ler_arquivo_texto(path_arquivo)
    data.append(texto)
    if subdir == 'pos':
      data_labels.append('positive')
    else:
      data_labels.append('negative')

df = pd.DataFrame({'review_text':np.array(data), 'sentiment':np.array(data_labels)})


## etapa 2 - normalização das entradas

import nltk # pip3 install nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
import emoji # pip3 install emoji
import re

from nltk.tokenize import word_tokenize
import string

nltk.download('punkt') 
nltk.download('stopwords')

def tokenize(review):
  tokens = word_tokenize(review)
  return tokens

def custom_tokenize(review):
  
  token_list = word_tokenize(review)

  token_list = [token for token in token_list
                  if token not in string.punctuation]

  token_list = [token for token in token_list if token.isalpha()]
  
  stop_words = set(stopwords.words('english'))
  stop_words.discard('not')
  token_list = [token for token in token_list if not token in stop_words]

  return token_list

def processar_texto(review):

  review = re.sub('(http|https):\/\/\S+', "", review)
  review = re.sub('#+', "", review)
  review = re.sub('\B@\w+', "", review)

  review = review.lower()
  review = re.sub(r'[\?\.\!]+(?=[\?\.\!])', "", review)
  review = re.sub(r'(.)\1+', r'\1\1', review)
  review = emoji.demojize(review)

  tokens = custom_tokenize(review)
  stemmer = SnowballStemmer("english")

  # stem tokens
  stem = []
  for token in tokens:
    stem.append(stemmer.stem(token))

  return stem

## etapa 3 - vetorização

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

df["tokens"] = df["review_text"].apply(processar_texto)
df["review_sentiment"] = df["sentiment"].apply(lambda i: 1
                                              if i == "positive" else 0)

X = df["tokens"].tolist()
y = df["review_sentiment"].tolist()

# Positive/Negative Frequency

def build_freqs(review_list, sentiment_list):
  freqs = {}
  for review, sentiment in zip(review_list, sentiment_list):
    for word in review:
      pair = (word, sentiment)
      if pair in freqs:
        freqs[pair] += 1
      else:
        freqs[pair] = 1
  return freqs

def review_to_freq(review, freqs):
  x = np.zeros((2,))
  for word in review:
    if (word, 1) in freqs:
      x[0] += freqs[(word, 1)]
    if (word, 0) in freqs:
      x[1] += freqs[(word, 0)]
  return x

# Bag of words

def fit_cv(review_corpus):
  cv_vect = CountVectorizer(tokenizer=lambda x: x,
                            preprocessor=lambda x: x)
  cv_vect.fit(review_corpus)
  return cv_vect

# Inverse Document Frequency (TF-IDF)

def fit_tfidf(review_corpus):
  tf_vect = TfidfVectorizer(preprocessor=lambda x: x,
                            tokenizer=lambda x: x)
  tf_vect.fit(review_corpus)
  return tf_vect

## modelos

import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0,
                                                    train_size=0.80)

# modelo 1 (pos/neg frequency)

freqs = build_freqs(X_train, y_train)
X_train_pn = [review_to_freq(review, freqs) for review in X_train]
X_test_pn = [review_to_freq(review, freqs) for review in X_test]

modelo1 = LogisticRegression()
modelo1.fit(X_train_pn, y_train)

# modelo 2 (bag of words)

cv = fit_cv(X_train)
X_train_cv = cv.transform(X_train)
X_test_cv = cv.transform(X_test)

modelo2 = LogisticRegression(solver='lbfgs', max_iter=1000) # solução para rodar o bag of words (https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter)
modelo2.fit(X_train_cv, y_train)

# modelo 3 (TF-IDF) + dump com joblib

tf = fit_tfidf(X_train)
X_train_tf = tf.transform(X_train)
X_test_tf = tf.transform(X_test)

modelo3 = LogisticRegression()
modelo3.fit(X_train_tf, y_train)
joblib.dump(modelo3, path_modelo+'modelo_celero.pkl')

## Métricas

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sn

def plotar_matriz(cm):
  plt.figure(figsize = (5,5))
  sn.heatmap(cm, annot=True, cmap="Greens", fmt='.0f')
  plt.xlabel("Predito")
  plt.ylabel("Verdadeiro")
  plt.title("Matriz de Confusão")
  plt.show() # no Google Collab vai só o "return n"
  return sn

# modelo 1

print("Precisão do modelo 1: {:.2%}".format(accuracy_score(y_test, modelo1.predict(X_test_pn))))
plotar_matriz(confusion_matrix(y_test, modelo1.predict(X_test_pn)))

# modelo 2

print("Precisão do modelo 2: {:.2%}".format(accuracy_score(y_test, modelo2.predict(X_test_cv))))
plotar_matriz(confusion_matrix(y_test, modelo2.predict(X_test_cv)))

# modelo 3

print("Precisão do modelo 3: {:.2%}".format(accuracy_score(y_test, modelo3.predict(X_test_tf))))
plotar_matriz(confusion_matrix(y_test, modelo3.predict(X_test_tf)))