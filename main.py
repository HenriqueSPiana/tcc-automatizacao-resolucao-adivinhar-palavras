from logica_w2v import obter_palavra_aleatoria
from automacao_selenium import escreve_palavra, enviar_palavra, acessa_contexto,navegador
global primeira_palavra


def main():
    global primeira_palavra
    primeira_palavra=obter_palavra_aleatoria()

    acessa_contexto(navegador)
    entrada = navegador.find_element("class name", "word")
    escreve_palavra(entrada,primeira_palavra)
    enviar_palavra(navegador)
    


if __name__ == "__main__":
    main()


