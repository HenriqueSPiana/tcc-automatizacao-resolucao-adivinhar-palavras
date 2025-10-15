import requests

class ContextoAPI:
    def __init__(self, dia: int):
        if not isinstance(dia, int) or dia <= 0:
            raise ValueError("O dia deve ser um número inteiro positivo.")
        self.dia = dia
        self.base_url = "https://api.contexto.me/machado/pt-br/game"

    def query(self, guess: str) -> int | None:
        """
        Envia 'guess' para a API e retorna o rank (distância).
        Se a palavra for inválida ou ocorrer um erro, retorna None.
        """
        url = f"{self.base_url}/{self.dia}/{guess.lower()}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return int(data.get('distance')) + 1
            return None
        except requests.RequestException as e:
            print(f"Erro de rede ao tentar chutar '{guess}': {e}")
            return None