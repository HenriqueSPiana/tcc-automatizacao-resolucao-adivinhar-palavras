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


def insere_palavra_e_envia(elemento, palavra):
    escreve_palavra(elemento, palavra)
    enviar_palavra()


def enviar_palavra():
    print("Enviando a palavra...")
    actions = ActionChains(navegador)
    actions.send_keys(Keys.RETURN).perform()
    time.sleep(3)

def validaInsercao():
    print("executando metodo validaInsercao")
    try:
        time.sleep(1)
        mensagem_elemento = navegador.find_element(By.CLASS_NAME, "message-text")
        print(mensagem_elemento.text)
        mensagem_esperada = ["Perdão, não conheço essa palavra","Essa palavra não vale porque é muito comum"]

        if(mensagem_esperada[0] == mensagem_elemento.text or mensagem_esperada[1] == mensagem_elemento.text):
            return False
        return True
    except Exception as e:
        print(f"Erro ao validar a palavra: {e}")
        return True

def limpa_campo():
    print("Limpando o campo...")
    campo_elemento = navegador.find_element(By.CLASS_NAME, "word")
    campo_elemento.clear()
    time.sleep(1);


def acessa_site():
    print("Acessando o site...")
    navegador.get("https://contexto.me")
    time.sleep(1)


def pega_palavra_navegador():
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
    