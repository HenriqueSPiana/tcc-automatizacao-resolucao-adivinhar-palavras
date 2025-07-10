from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

import time






def escrevePalavra(elemento,palavra):
        elemento.clear()
        elemento.send_keys(palavra)

def enviarPalavra(driver):
    actions = ActionChains(driver)
    actions.send_keys(Keys.RETURN).perform()


    


def main():

    primeiraPalavra="casa"
    navegador = webdriver.Chrome()
    navegador.get("https://contexto.me")
    time.sleep(5)
    input = navegador.find_element("class name", "word")

    escrevePalavra(input,primeiraPalavra)
    time.sleep(3)
    enviarPalavra(navegador)
    time.sleep(3)


if __name__ == "__main__":
    main()


