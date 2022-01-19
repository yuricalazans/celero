# Bibliotecas "não nativas" utilizadas e comandos de instalação (PIP)

sklearn - pip3 install scikit-learn

flask - pip3 install flask 

pandas - pip3 install pandas

matplotlib - pip3 install matplotlib

seaborn - pip3 install seaborn

numpy - pip3 install numpy

nltk - pip3 install nltk

emoji - pip3 install emoji

re - pip3 install re

# Modo de uso - Configuração 1 (treinamento dos modelos e execução via API)

python3 celero.py -pt "/.../aclImdb/train/" -pm "/.../celero"

# Modo de uso - Configuração 2 (consumo da API via curl)
curl -X POST -i 'http://127.0.0.1:81/' --data-binary "@review.txt"
