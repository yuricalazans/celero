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