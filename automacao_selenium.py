from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from logica_w2v import adiciona_na_pilha;
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
    print("executando metodo isPalavra")
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
    
    print("Palavra válida.")
    return True

def acessa_site(navegador):
    print("Acessando o site...")
    navegador.get("https://contexto.me")
    time.sleep(1)


def pega_palavra_navegador(navegador):
    print("Pegando a palavra...")
    css_selector = "div.row span:nth-child(1)"
    first_span = navegador.find_element(By.CSS_SELECTOR, css_selector)
    adiciona_na_pilha(str(first_span.text))


if __name__ == '__main__':
    acessa_site(navegador)
    entrada = navegador.find_element("class name", "word")
    escreve_palavra(entrada, "teste")
    enviar_palavra(navegador)
    print(pega_palavra_navegador(navegador))
    