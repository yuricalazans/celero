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