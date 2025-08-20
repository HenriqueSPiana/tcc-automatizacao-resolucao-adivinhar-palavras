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
    time.sleep(1)

def isPalavra():
    print("Validando a palavra...")
    try:
        WebDriverWait(navegador, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "message-text"))
        )
        return True
    except Exception as e:
        print(f"Erro ao validar a palavra: {e}")
        return False


def valida_palavra():
    print("Validando a palavra...")
    if not isPalavra():
        obter_palavra_aleatoria()
        valida_palavra()
    else:
        print("Palavra válida.")

def acessa_site(navegador):
    print("Acessando o site...")
    navegador.get("https://contexto.me")
    time.sleep(1)


def pega_aderencia(navegador):
    print("Pegando a aderência...")
    css_selector = "div.row span:nth-child(2)"
    second_span = navegador.find_element(By.CSS_SELECTOR, css_selector)
    return int(second_span.text)


def calcula_aderencia(metodo,navegador,entrada):
    aderencia = pega_aderencia(navegador)
    print(f"Aderência: {aderencia}")
    if (aderencia>=3000):
        print("Aderência muito alta pegando outra palavra")
        escreve_palavra(entrada, metodo())
        enviar_palavra(navegador)
        print(pega_aderencia(navegador))
        calcula_aderencia(metodo,navegador,entrada);
    print("Aderência abaixo de 3000")

if __name__ == '__main__':
    acessa_site(navegador)
    entrada = navegador.find_element("class name", "word")
    escreve_palavra(entrada, "teste")
    enviar_palavra(navegador)
    print(pega_aderencia(navegador))
    calcula_aderencia(obter_palavra_aleatoria,navegador,entrada)
    