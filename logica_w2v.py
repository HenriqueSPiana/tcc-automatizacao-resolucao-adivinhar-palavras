import gensim
import random
import os
import nltk
import zipfile

NOME_ARQUIVO_MODELO = 'cbow_s300.txt' 
modelo_carregado = None



def descompactar_arquivo(zip_path, extrair_para):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extrair_para)


def carregar_modelo():

    global modelo_carregado

    if modelo_carregado is None:
        if not os.path.exists(NOME_ARQUIVO_MODELO):
            descompactar_arquivo('cbow_s300.zip', './modelos')

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
    baixa_dicionario()
    tags_validas = ('N', 'V', 'ADJ')
    palavras_validas = []
    # 'N', 'V', 'ADJ' significam substantivo, verbo e adjetivo, respectivamente.

    dicionario = nltk.corpus.mac_morpho.tagged_words()

    for palavra, tag in dicionario:
        # Se a tag começa com uma das válidas e se ela é uma palavra válida, o isalpha elimina caracteres especiais e numeros, 
        # deixando somente palavras sem hifen e sem espaços
        if tag.startswith(tags_validas) and palavra.isalpha():
            palavras_validas.append(palavra)

    palavra_aleatoria = random.choice(palavras_validas).lower()
    return palavra_aleatoria


def obter_palavra_similar(palavra):
    carregar_modelo()
    return modelo_carregado.most_similar(positive=palavra, topn=1)



    
    
    



if __name__ == '__main__':
    print("Testando a função de obter palavra aleatória...")


    obter_palavra_similar("teste")
    # for i in range(5):
    #     print(f"\nA palavra aleatória escolhida é: '{obter_palavra_aleatoria()}'")