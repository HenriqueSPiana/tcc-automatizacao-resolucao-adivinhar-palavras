from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time
from logica_w2v import obter_palavra_aleatoria


def escreve_palavra(elemento,palavra):
        elemento.clear()
        elemento.send_keys(palavra)

def enviar_palavra(driver):
    actions = ActionChains(driver)
    actions.send_keys(Keys.RETURN).perform()

def main():

    primeira_palavra=obter_palavra_aleatoria()
    navegador = webdriver.Chrome()
    navegador.get("https://contexto.me")
    time.sleep(5)
    entrada = navegador.find_element("class name", "word")

    escreve_palavra(entrada,primeira_palavra)
    time.sleep(3)
    enviar_palavra(navegador)
    time.sleep(3)


if __name__ == "__main__":
    main()


