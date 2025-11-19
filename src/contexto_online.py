from playwright.sync_api import sync_playwright, TimeoutError

class ContextoOnline:
    def __init__(self, headless: bool = True):
        """
        headless: True roda sem janela
        """
        self.headless = headless
        self._play = None
        self._browser = None
        self._page = None

    def start(self):
        self._play = sync_playwright().start()
        self._browser = self._play.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page()
        self._page.goto("https://contexto.me/pt/", wait_until="domcontentloaded", timeout=60000)
        return self
    
    def close(self):
        if self._browser:
            self._browser.close()

        if self._play:
            self._play.stop()

    def query(self, guess: str) -> int | None:
        """
        Envia 'guess' e retorna o rank (int). Se não conseguir ler, retorna None.
        Corrigido para ler a PRIMEIRA linha (a mais recente) e extrair números com milhares/sufixos.
        """
        # Envia chute
        print(f"Enviando chute: {guess}")
        self._page.fill("input[type='text'], input", guess)
        self._page.keyboard.press("Enter")

        try:
            self._page.wait_for_selector("div.loading-text", state="hidden", timeout=5000)
        except Exception:
            pass
        
        if self._page.query_selector("div.message-text") is not None:
            return None

        # Lê a PRIMEIRA linha (mais recente)
        rows = self._page.query_selector_all("div.row")
        if not rows:
            return None
        row = rows[0]

        # Extrai a palavra e o rank diretamente dos spans
        spans = row.query_selector_all("span")
        rank = int(spans[1].inner_text().strip())
        
        return rank