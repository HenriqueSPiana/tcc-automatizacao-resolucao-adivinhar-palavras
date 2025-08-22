from logica_w2v import obter_palavra_aleatoria,obter_palavra_similar,carregar_modelo,pega_palavra_pilha,adiciona_na_pilha
from automacao_selenium import *



def main():
    adiciona_na_pilha('teshadste')
    acessa_site(navegador)
    campo_entrada = navegador.find_element("class name", "word")
    # obter_palavra_aleatoria()
    escreve_palavra(campo_entrada,pega_palavra_pilha())
    enviar_palavra(navegador)
    while not(valida_palavra(navegador)):
        obter_palavra_aleatoria()
        ultima_palavra = pega_palavra_pilha()
        escreve_palavra(campo_entrada,ultima_palavra)
        enviar_palavra(navegador)
    carregar_modelo()
    obter_palavra_similar(navegador,ultima_palavra)
    ultima_palavra = pega_palavra_pilha()
    escreve_palavra(campo_entrada,ultima_palavra)
    enviar_palavra(navegador)




if __name__ == "__main__":
    main()