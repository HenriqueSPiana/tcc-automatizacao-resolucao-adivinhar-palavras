from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from logica_w2v import obter_palavra_aleatoria;
import time
global navegador
navegador = webdriver.Chrome()


def escreve_palavra(elemento,palavra):
    elemento.clear()
    elemento.send_keys(palavra)
    time.sleep(3)


def enviar_palavra(driver):
    actions = ActionChains(driver)
    actions.send_keys(Keys.RETURN).perform()
    time.sleep(3)


def acessa_contexto(navegador):
    navegador.get("https://contexto.me")
    time.sleep(3)


def pega_adesao(navegador):
    css_selector = "div.row span:nth-child(2)"
    second_span = navegador.find_element(By.CSS_SELECTOR, css_selector)
    return int(second_span.text)


def calcula_adesao(metodo,navegador,entrada):
    adesao = pega_adesao(navegador)
    if (adesao>=3000):
        escreve_palavra(entrada, metodo())
        enviar_palavra(navegador)
        print(pega_adesao(navegador))
        calcula_adesao(metodo,navegador,entrada);
    else:
        print("Adesão abaixo de 3000, encerrando.")

if __name__ == '__main__':
    acessa_contexto(navegador)
    entrada = navegador.find_element("class name", "word")
    escreve_palavra(entrada, "teste")
    enviar_palavra(navegador)
    print(pega_adesao(navegador))
    calcula_adesao(obter_palavra_aleatoria,navegador,entrada)    

