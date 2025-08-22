from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from logica_w2v import obter_palavra_aleatoria;
import time
global navegador
navegador = webdriver.Chrome()


def escreve_palavra(elemento,palavra):
    elemento.clear()
    print(f"Escrevendo a palavra: {palavra}")
    elemento.send_keys(palavra)
    time.sleep(1)


def enviar_palavra(driver):
    print("Enviando a palavra...")
    actions = ActionChains(driver)
    actions.send_keys(Keys.RETURN).perform()
    time.sleep(3)

def isPalavra(navegador):
    print("Validando a palavra...")
    try:
        time.sleep(1)
        mensagem_elemento = navegador.find_element(By.CLASS_NAME, "message-text")
        print(mensagem_elemento.text)
        mensagem_esperada = "Perdão, não conheço essa palavra"

        if(mensagem_esperada == mensagem_elemento.text):
            return False
        return True
    except Exception as e:
        print(f"Erro ao validar a palavra: {e}")
        return True

def limpa_campo(navegador):
    print("Limpando o campo...")
    campo_elemento = navegador.find_element(By.CLASS_NAME, "word")
    campo_elemento.clear()
    time.sleep(1);


def valida_palavra(navegador):
    print("Validando a palavra...")
    if not isPalavra(navegador):
        print("Palavra invalida.")
        limpa_campo(navegador)
        return False
    else:
        print("Palavra válida.")
        return True

def acessa_site(navegador):
    print("Acessando o site...")
    navegador.get("https://contexto.me")
    time.sleep(1)


def pega_aderencia(navegador):
    print("Pegando a aderência...")
    css_selector = "div.row span:nth-child(2)"
    second_span = navegador.find_element(By.CSS_SELECTOR, css_selector)
    return int(second_span.text)


if __name__ == '__main__':
    acessa_site(navegador)
    entrada = navegador.find_element("class name", "word")
    escreve_palavra(entrada, "teshadste")
    enviar_palavra(navegador)
    # print(pega_aderencia(navegador))
    # calcula_aderencia(obter_palavra_aleatoria,navegador,entrada)
    