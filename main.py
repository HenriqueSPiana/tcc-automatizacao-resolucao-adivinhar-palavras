import random
import numpy as np
from typing import Callable, List, Tuple, Set
import nltk
from sklearn.linear_model import Ridge
from gensim.models import KeyedVectors
from simplemma import lemmatize
import requests 
import concurrent.futures
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # <-- NOVA IMPORTAÇÃO para criar diretórios

# ===================================================================
# SUAS CLASSES
# ===================================================================

class ContextoAPI:
    # ... (código da classe sem alterações)
    BASE_URL = "https://api.contexto.me/machado/pt-br/game"
    def __init__(self, dia: int):
        if not isinstance(dia, int) or dia <= 0:
            raise ValueError("O dia deve ser um número inteiro positivo.")
        self.dia = dia
    def query(self, guess: str) -> int | None:
        url = f"{self.BASE_URL}/{self.dia}/{guess.lower()}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('distance')
            return None
        except requests.RequestException:
            return None

class ContextoSolver:
    def __init__(
        self,
        model: KeyedVectors,
        query_function: Callable,
        optimization_rank_limit: int = 100,
        max_vocab: int = 150_000,
        random_state: int = 42,
        log_file: str | None = None  # <-- NOVO PARÂMETRO OPCIONAL
    ):
        self.model = model
        self.query_function = query_function
        self.optimization_rank_limit = optimization_rank_limit
        self.rng = random.Random(random_state)
        self.log_file = log_file  # <-- ARMAZENA O NOME DO ARQUIVO DE LOG
        
        # O resto do __init__ permanece o mesmo...
        with open("dicionario_ptbr.txt", "r", encoding="utf-8") as f:
            self.portuguese_dictionary = {line.strip().lower() for line in f}
        self.stopwords = set(nltk.corpus.stopwords.words("portuguese"))
        base_vocab = model.index_to_key[:max_vocab]
        self.vocab = {w for w in base_vocab if self._is_valid_word(w)}
        default_seeds = ["animal", "objeto", "lugar", "corpo", "natureza", "alimento", "ferramenta", "sentimento"]
        self.seed_words = [self._apply_lemmatizer(w) for w in default_seeds if w in self.vocab]
        self.guessed = set()
        self.history = list()
        self.best_words = list()
        self.best_rank = float("inf")
        self.reg = Ridge(alpha=0.5, random_state=random_state)
        self.X_obs = list()
        self.y_obs = list()
        self._fitted = False 
        self.optimization_mode = False

    # ... (demais métodos da classe sem alterações) ...
    def _get_batch_candidates(self, batch_size: int) -> List[str]:
        if len(self.history) < 5:
            seeds = [s for s in self.seed_words if self._is_new(s)]
            return seeds[:batch_size]
        raw_candidates = self._get_candidates_from_history()
        lemma_candidates = {self._apply_lemmatizer(c) for c in raw_candidates}
        valid_lemmas = [lemma for lemma in lemma_candidates if lemma in self.vocab and self._is_new(lemma)]
        if not valid_lemmas:
            unguessed = [w for w in list(self.vocab) if self._is_new(w)]
            return self.rng.sample(unguessed, min(len(unguessed), batch_size))
        if self._fitted:
            cand_vecs = np.vstack([self.model.get_vector(w, norm=True) for w in valid_lemmas])
            preds = self.reg.predict(cand_vecs)
            best_indices = np.argsort(preds)[::-1][:batch_size]
            return [valid_lemmas[i] for i in best_indices]
        else:
            top_3_vecs = [self.model.get_vector(w, norm=True) for w, r in self.best_words[:3]]
            scores = []
            for cand in valid_lemmas:
                cand_vec = self.model.get_vector(cand, norm=True)
                avg_sim = sum(np.dot(cand_vec, top_vec) for top_vec in top_3_vecs) / len(top_3_vecs)
                scores.append((avg_sim, cand))
            scores.sort(key=lambda x: x[0], reverse=True)
            return [cand for score, cand in scores[:batch_size]]
    def _observe_and_learn(self, word: str, rank: int):
        self.history.append((word, rank))
        self.guessed.add(self._apply_lemmatizer(word))
        self.best_words.append((word, rank))
        self.best_words.sort(key=lambda x: x[1])
        if rank < self.best_rank:
            self.best_rank = rank
            if not self.optimization_mode and self.best_rank < self.optimization_rank_limit:
                self.optimization_mode = True
        if self.optimization_mode:
            if rank > 0:
                y = 1.0 / np.log1p(rank)
                self.X_obs.append(self.model.get_vector(word, norm=True))
                self.y_obs.append(y)
                if len(self.X_obs) >= 15:
                    self.reg.fit(np.vstack(self.X_obs), np.array(self.y_obs))
                    self._fitted = True
    def _get_candidates_from_history(self) -> Set[str]:
        pool = set()
        if not self.best_words: return pool
        best_word, _ = self.best_words[0]
        for w, _ in self.model.most_similar(best_word, topn=50):
            if w in self.vocab: pool.add(w)
        if len(self.best_words) < 3: return pool
        top_5 = [w for w, r in self.best_words[:5]]
        pivot_worst = self.best_words[min(len(self.best_words) - 1, 10)][0]
        if len(top_5) >= 2:
            for w, _ in self.model.most_similar(positive=top_5[:2], topn=20):
                if w in self.vocab: pool.add(w)
        for good_word in top_5:
            if good_word == pivot_worst: continue
            try:
                for target_word in top_5[:3]:
                    if target_word == good_word: continue
                    for w, _ in self.model.most_similar(positive=[target_word, good_word], negative=[pivot_worst], topn=10):
                        if w in self.vocab: pool.add(w)
            except KeyError: continue
        return pool

    def solve(self, max_attempts: int = 200, verbose: bool = True, batch_size: int = 8):
        attempt_num = 1
        while attempt_num <= max_attempts:
            batch_to_guess = self._get_batch_candidates(batch_size)
            if not batch_to_guess: break
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_word = {executor.submit(self.query_function, word): word for word in batch_to_guess}
                for future in concurrent.futures.as_completed(future_to_word):
                    word = future_to_word[future]
                    try:
                        rank = future.result()
                        if rank is not None: results.append({'word': word, 'rank': rank})
                        else: self.guessed.add(self._apply_lemmatizer(word))
                    except Exception: pass
            if not results:
                attempt_num += len(batch_to_guess)
                continue
            results.sort(key=lambda x: x['rank'])
            found_in_batch = False
            for result in results:
                word, rank = result['word'], result['rank']
                if self._is_new(word): self._observe_and_learn(word, rank)
                else: continue
                
                # --- LÓGICA DE LOG CORRIGIDA ---
                # Define o 'mode' ANTES de usá-lo.
                mode = "OPTIMIZE" if self.optimization_mode else "EXPLORE" # <-- MUDANÇA: Linha movida para cá
                
                # Agora cria a mensagem de log, com 'mode' já definido.
                log_message = f"Tentativa {attempt_num:03d} [{mode}]: {word:<15} → rank {rank:<5} (Melhor: {self.best_rank})"
                
                # Se verbose=True, imprime no console.
                if verbose:
                    print(f"{log_message} [Dia: {self.query_function.__self__.dia}]")

                # Se um arquivo de log foi especificado, salva a mensagem nele.
                if self.log_file:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(log_message + '\n')
                # --- FIM DA LÓGICA CORRIGIDA ---

                attempt_num += 1
                if rank == 0:
                    found_in_batch = True
                    break
            if found_in_batch: break
        return self.history

    def _is_valid_word(self, w: str) -> bool:
        return (w.isalpha() and w.islower() and w not in self.stopwords and len(w) > 2 and w in self.portuguese_dictionary)
    def _apply_lemmatizer(self, w: str) -> str:
        return lemmatize(w, lang="pt")
    def _is_new(self, w: str) -> bool:
        return self._apply_lemmatizer(w) not in self.guessed

# ===================================================================
# FUNÇÕES DE EXPERIMENTO E VISUALIZAÇÃO
# ===================================================================

def run_experiment(model, game_range: range, max_attempts: int):
    """
    Roda o solver para uma série de jogos e coleta dados detalhados.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    all_results = []
    
    for dia in game_range:
        print(f"\n--- Iniciando Jogo do Dia: {dia} ---")
        start_time = time.time()
        
        log_filename = os.path.join(log_dir, f"jogo_{dia}.log")
        contexto = ContextoAPI(dia=dia)
        solver = ContextoSolver(model, query_function=contexto.query, log_file=log_filename)
        
        history = solver.solve(max_attempts=max_attempts, verbose=False)
        
        end_time = time.time()

        # --- Coleta de Métricas Aprimorada ---
        status = "SUCESSO" if solver.best_rank == 0 else "FALHA"
        tentativas_totais = len(history)
        
        # <<< MUDANÇA AQUI: Captura a palavra final e o rank exato >>>
        palavra_final = ""
        rank_final = -1
        if solver.best_words: # Garante que a lista não está vazia
            palavra_final = solver.best_words[0][0] # Pega a palavra com o melhor rank
            rank_final = solver.best_words[0][1] # Pega o melhor rank (será 0 ou 1 no sucesso)

        # Calcula tentativas de exploração vs otimização
        tentativas_exploracao = tentativas_totais
        for i, (word, rank) in enumerate(history):
            if rank < solver.optimization_rank_limit:
                tentativas_exploracao = i + 1
                break
        
        tentativas_otimizacao = tentativas_totais - tentativas_exploracao

        result = {
            "id_jogo": dia,
            "status": status,
            "palavra_final": palavra_final, # <-- NOVO DADO
            "rank_final": rank_final,       # <-- NOVO DADO
            "tentativas_totais": tentativas_totais,
            "tempo_total_s": round(end_time - start_time, 2),
            "tentativas_exploracao": tentativas_exploracao,
            "tentativas_otimizacao": tentativas_otimizacao
        }
        all_results.append(result)
        
        print(f"--- Fim do Jogo {dia}: {status} com '{palavra_final}' (rank {rank_final}) em {tentativas_totais} tentativas ---")

    df = pd.DataFrame(all_results)
    df.to_csv("resultados_contexto.csv", index=False)
    print("\nResultados do experimento salvos em 'resultados_contexto.csv'")
    return df

def analyze_and_plot(df: pd.DataFrame):
    """
    Lê o DataFrame com os resultados e gera os gráficos, incluindo a análise de falhas.
    """
    if df.empty:
        print("O DataFrame de resultados está vazio. Nenhum gráfico será gerado.")
        return

    # Configura o estilo dos gráficos
    sns.set_theme(style="whitegrid")
    
    # Cria uma figura para conter os 4 gráficos (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11)) # Aumentei um pouco o tamanho para melhor visualização
    fig.suptitle('Análise de Desempenho do ContextoSolver', fontsize=18)

    # --- Gráfico 1: Taxa de Sucesso (Pizza) ---
    status_counts = df['status'].value_counts()
    if 'SUCESSO' not in status_counts: status_counts['SUCESSO'] = 0
    if 'FALHA' not in status_counts: status_counts['FALHA'] = 0
    axes[0, 0].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
                   colors=['#4CAF50', '#F44336'], startangle=90)
    axes[0, 0].set_title('Taxa de Sucesso Geral', fontsize=14)
    axes[0, 0].set_ylabel('')

    # --- Gráfico 2: Distribuição de Tentativas (Histograma) ---
    df_success = df[df['status'] == 'SUCESSO']
    sns.histplot(data=df_success, x='tentativas_totais', kde=True, ax=axes[0, 1], bins=15)
    axes[0, 1].set_title('Distribuição de Tentativas para Sucessos', fontsize=14)
    axes[0, 1].set_xlabel('Número de Tentativas')
    axes[0, 1].set_ylabel('Contagem de Jogos')

    # --- Gráfico 3: Distribuição do Tempo de Execução (Histograma) ---
    sns.histplot(data=df_success, x='tempo_total_s', kde=True, ax=axes[1, 0], color='skyblue', bins=15)
    axes[1, 0].set_title('Distribuição do Tempo de Execução para Sucessos', fontsize=14)
    axes[1, 0].set_xlabel('Tempo Total (segundos)')
    axes[1, 0].set_ylabel('Contagem de Jogos')

    # ===================================================================================
    # --- GRÁFICO 4 (NOVO): Distribuição do Melhor Rank nas Falhas ---
    # ===================================================================================
    df_failure = df[df['status'] == 'FALHA']
    
    if not df_failure.empty:
        # Usamos uma escala de log no eixo X para visualizar melhor a grande variação de ranks
        sns.histplot(data=df_failure, x='rank_final', ax=axes[1, 1], color='#FF8A65', log_scale=True, kde=False)
        axes[1, 1].set_title('Distribuição do Rank Final para Falhas', fontsize=14)
        axes[1, 1].set_xlabel('Melhor Rank Alcançado (Escala Logarítmica)')
        axes[1, 1].set_ylabel('Contagem de Jogos')
    else:
        # Caso não haja nenhuma falha, exibe uma mensagem no gráfico
        axes[1, 1].text(0.5, 0.5, 'Nenhuma falha registrada', 
                       horizontalalignment='center', verticalalignment='center', 
                       fontsize=14, color='gray')
        axes[1, 1].set_title('Distribuição do Rank Final para Falhas', fontsize=14)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])


    # Ajusta o layout para evitar sobreposição e exibe o painel
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # ... (bloco __main__ sem alterações)
    nltk.download('stopwords', quiet=True)
    nltk.download('rslp', quiet=True)
    print("Carregando modelo Word2Vec (pode levar um tempo)...")
    model = KeyedVectors.load_word2vec_format("cbow_s300.txt", binary=False)
    print("Modelo carregado com sucesso.")
    JOGOS_A_TESTAR = range(1, 20)
    MAX_TENTATIVAS_POR_JOGO = 400
    df_resultados = run_experiment(model, JOGOS_A_TESTAR, MAX_TENTATIVAS_POR_JOGO)
    analyze_and_plot(df_resultados)
    print("\n--- Análise completa ---")