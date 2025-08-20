from logica_w2v import obter_palavra_aleatoria,obter_palavra_similar,carregar_modelo
from automacao_selenium import *
global primeira_palavra


def main():
    carregar_modelo()
    global primeira_palavra
    primeira_palavra=obter_palavra_aleatoria()
    acessa_contexto(navegador)
    entrada = navegador.find_element("class name", "word")
    escreve_palavra(entrada,primeira_palavra)
    enviar_palavra(navegador)
    print(pega_aderencia(navegador))
    calcula_aderencia(obter_palavra_aleatoria,navegador,entrada)
    print(obter_palavra_similar(primeira_palavra))


if __name__ == "__main__":
    main()