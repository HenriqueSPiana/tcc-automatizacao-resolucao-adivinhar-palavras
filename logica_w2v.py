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


def baixa_dicionario():
   if not nltk.data.find('corpora/mac_morpho'):
        print("Corpus 'mac_morpho' não encontrado, baixando...")
        nltk.download('mac_morpho')
        print("Corpus 'mac_morpho' baixado com sucesso.")  



def obter_palavra_aleatoria():
    tags_validas = ('N', 'V', 'ADJ')
    palavra_aleatoria = []
    # 'N', 'V', 'ADJ' significam substantivo, verbo e adjetivo, respectivamente.

    dicionario = nltk.corpus.mac_morpho.tagged_words()

    for palavra, tag in dicionario:
        # Se a tag começa com uma das válidas
        if tag.startswith(tags_validas):
            palavra_aleatoria.append(palavra)

    return random.choice(palavra_aleatoria).lower()

if __name__ == '__main__':
    print("Testando a função de obter palavra aleatória...")

    for i in range(5):
        print(f"\nA palavra aleatória escolhida é: '{obter_palavra_aleatoria()}'")