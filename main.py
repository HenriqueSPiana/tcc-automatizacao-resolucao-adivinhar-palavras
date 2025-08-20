from logica_w2v import obter_palavra_aleatoria,obter_palavra_similar,carregar_modelo,pega_palavra_pilha
from automacao_selenium import *



def main():
    carregar_modelo()
    acessa_site(navegador)
    campo_entrada = navegador.find_element("class name", "word")
    obter_palavra_aleatoria()
    escreve_palavra(campo_entrada,pega_palavra_pilha())
    enviar_palavra(navegador)
    valida_palavra()
    calcula_aderencia(obter_palavra_aleatoria,navegador,campo_entrada)



if __name__ == "__main__":
    main()