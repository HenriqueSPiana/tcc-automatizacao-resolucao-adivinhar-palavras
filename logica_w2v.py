import gensim
import random
import os
import nltk

NOME_ARQUIVO_MODELO = 'cbow_s300.txt' 
modelo_carregado = None
vocabulario_filtrado = None

def carregar_modelo():

    global modelo_carregado

    if modelo_carregado is None:
        if not os.path.exists(NOME_ARQUIVO_MODELO):
            raise FileNotFoundError(
                f"Arquivo do modelo '{NOME_ARQUIVO_MODELO}' não encontrado. "
            )
        
        print("Carregando modelo")
        modelo_carregado = gensim.models.KeyedVectors.load_word2vec_format(
            NOME_ARQUIVO_MODELO, 
            binary=False
        )
        print("Modelo carregado")
        
    return modelo_carregado

def obter_palavra_aleatoria():
    try:
        nltk.data.find('corpora/mac_morpho')
    except LookupError:
        print("Corpus 'mac_morpho' não encontrado")
        nltk.download('mac_morpho')
        print("Download concluído. ")

    dicionario = nltk.corpus.mac_morpho.words()
    palavra_aleatoria = random.choice(dicionario)
    while len(palavra_aleatoria) < 5:
        palavra_aleatoria = random.choice(dicionario)

    return palavra_aleatoria.lower()

if __name__ == '__main__':
    print("Testando a função de obter palavra aleatória...")

    for i in range(5):
        print(f"\nA palavra aleatória escolhida é: '{obter_palavra_aleatoria()}'")