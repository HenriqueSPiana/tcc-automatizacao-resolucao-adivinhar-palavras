from logica_w2v import obter_palavra_aleatoria,obter_palavra_similar,carregar_modelo,pega_palavra_pilha,adiciona_na_pilha
from automacao_selenium import *



def main():
    #proxima vez que for testar, lembrar que é necessario ter conexão com a web
    obter_palavra_aleatoria()
#    adiciona_na_pilha('teshadste')
    acessa_site(navegador)
    campo_entrada = navegador.find_element("class name", "word")
    insere_palavra_e_envia(campo_entrada,pega_palavra_pilha());
    while not(valida_palavra(navegador)):
        obter_palavra_aleatoria()
        insere_palavra_e_envia(campo_entrada,pega_palavra_pilha());
    carregar_modelo()
    pega_palavra_navegador(navegador)
    obter_palavra_similar(pega_palavra_pilha())
    insere_palavra_e_envia(campo_entrada,pega_palavra_pilha())




if __name__ == "__main__":
    main()