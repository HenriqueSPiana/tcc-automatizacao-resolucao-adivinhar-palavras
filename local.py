import random
import numpy as np
from typing import Callable, List, Tuple, Optional, Set
import nltk
from sklearn.linear_model import Ridge
from gensim.models import KeyedVectors

# --- INÍCIO: Seção de download automático do NLTK ---
# Esta nova seção garante que os dados necessários estejam presentes.
try:
    from nltk.corpus import stopwords
    STOP_PT: Set[str] = set(stopwords.words("portuguese"))
except (LookupError, OSError):
    print("Pacote 'stopwords' do NLTK não encontrado. Baixando agora...")
    # quiet=False para mostrar o progresso do download ao usuário
    nltk.download('stopwords', quiet=False)
    from nltk.corpus import stopwords
    STOP_PT: Set[str] = set(stopwords.words("portuguese"))
    print("Download concluído.")
# --- FIM: Seção de download automático do NLTK ---


# --- Seção de carregamento do dicionário pt-BR ---
def load_dictionary(file_path: str) -> Set[str]:
    """Carrega palavras de um arquivo de texto para um set, para verificação rápida."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return {line.strip().lower() for line in f}
    except FileNotFoundError:
        print("-" * 50)
        print(f"ERRO: Arquivo de dicionário não encontrado em '{file_path}'")
        print("Por favor, baixe o arquivo 'dicionario_ptbr.txt' e coloque-o na mesma pasta do script.")
        print("Link para download: https://raw.githubusercontent.com/python-br/palavras/master/palavras.txt")
        print("-" * 50)
        exit()

# Carrega o dicionário pt-BR.
PTBR_DICTIONARY = load_dictionary("dicionario_ptbr.txt")


# --- Função de validação de palavra ATUALIZADA ---
def is_valid_word(w: str) -> bool:
    """
    Verifica se a palavra é válida:
    1. Apenas letras.
    2. Minúscula.
    3. Não é uma stopword.
    4. Tem mais de 2 letras.
    5. EXISTE NO DICIONÁRIO PT-BR.
    """
    return (
        w.isalpha() and
        w.islower() and
        w not in STOP_PT and
        len(w) > 2 and
        w in PTBR_DICTIONARY
    )

class OfflineContextoOracle:
    # (Esta classe permanece inalterada)
    def __init__(self, model: KeyedVectors, secret: str):
        if secret not in model:
            raise ValueError(f"A palavra secreta '{secret}' não está no vocabulário do modelo.")
        self.model = model
        self.secret = secret
        self.rank_map = self._precompute_ranks()

    def _precompute_ranks(self) -> dict:
        secret_vec = self.model.get_vector(self.secret, norm=True)
        M = self.model.get_normed_vectors()
        sims = M @ secret_vec
        order = np.argsort(-sims)
        rank_map = {self.model.index_to_key[idx]: pos for pos, idx in enumerate(order, start=1)}
        return rank_map

    def query_rank(self, guess: str) -> Optional[int]:
        return self.rank_map.get(guess, None)

class HybridContextoSolver:
    # (Esta classe permanece praticamente inalterada, mas agora se beneficia da validação mais forte)
    def __init__(
        self,
        model: KeyedVectors,
        oracle_query: Callable[[str], Optional[int]],
        optimization_rank_threshold: int = 100,
        max_vocab: int = 150_000,
        random_state: int = 42,
    ):
        self.model = model
        self.oracle_query = oracle_query
        self.optimization_rank_threshold = optimization_rank_threshold
        self.rng = random.Random(random_state)

        print("Filtrando vocabulário com base no dicionário (pode levar um momento)...")
        base_vocab = model.index_to_key[:max_vocab]
        self.vocab = {w for w in base_vocab if is_valid_word(w)}
        print(f"Vocabulário filtrado para {len(self.vocab)} palavras válidas.")

        default_seeds = ["animal", "objeto", "lugar", "corpo", "natureza", "alimento", "ferramenta", "sentimento"]
        self.seed_words = [w for w in default_seeds if w in self.vocab]

        self.guessed: Set[str] = set()
        self.history: List[Tuple[str, int]] = []
        self.best_words: List[Tuple[str, int]] = []
        self.best_rank = float("inf")

        self.reg = Ridge(alpha=0.5, random_state=random_state)
        self.X_obs: List[np.ndarray] = []
        self.y_obs: List[float] = []
        self._fitted = False
        self.optimization_mode = False

    def _observe_and_learn(self, word: str, rank: int):
        self.history.append((word, rank))
        self.guessed.add(word)
        self.best_words.append((word, rank))
        self.best_words.sort(key=lambda x: x[1])
        
        if rank < self.best_rank:
            self.best_rank = rank
            if not self.optimization_mode and self.best_rank < self.optimization_rank_threshold:
                self.optimization_mode = True
                print(f"\n--- ATIVANDO MODO DE OTIMIZAÇÃO (MELHOR RANK: {self.best_rank}) ---\n")

        if self.optimization_mode:
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

    def _choose_next(self) -> str:
        if len(self.history) < 5:
            for seed in self.seed_words:
                if seed not in self.guessed: return seed
            return self.rng.choice([w for w in list(self.vocab) if w not in self.guessed])

        candidates = self._get_candidates_from_history()
        valid_candidates = [c for c in candidates if c not in self.guessed]

        if self._fitted and valid_candidates:
            cand_vecs = np.vstack([self.model.get_vector(w, norm=True) for w in valid_candidates])
            preds = self.reg.predict(cand_vecs)
            return valid_candidates[np.argmax(preds)]

        if valid_candidates:
            top_3_vecs = [self.model.get_vector(w, norm=True) for w,r in self.best_words[:3]]
            best_cand, max_sim = None, -2.0
            for cand in valid_candidates:
                cand_vec = self.model.get_vector(cand, norm=True)
                avg_sim = sum(np.dot(cand_vec, top_vec) for top_vec in top_3_vecs) / len(top_3_vecs)
                if avg_sim > max_sim: max_sim, best_cand = avg_sim, cand
            return best_cand
        
        return self.rng.choice([w for w in list(self.vocab) if w not in self.guessed])

    def solve(self, max_attempts: int = 200, verbose: bool = True) -> List[Tuple[str, int]]:
        for attempt in range(1, max_attempts + 1):
            current = self._choose_next()
            if current in self.guessed: continue

            rank = self.oracle_query(current)
            if rank is None:
                self.guessed.add(current)
                continue

            self._observe_and_learn(current, rank)
            
            if verbose:
                mode = "OPTIMIZE" if self.optimization_mode else "EXPLORE"
                print(f"Tentativa {attempt:03d} [{mode}]: {current:<15} → rank {rank:<5} (Melhor: {self.best_rank})")

            if rank == 1:
                print(f"\n✅ Descoberta em {attempt} tentativas: {current}")
                break
        
        if self.best_rank != 1:
            print(f"\n❌ Não encontrou a palavra. Melhor tentativa: '{self.best_words[0][0]}' (rank {self.best_words[0][1]})")

        return self.history

# --- Execução ---
try:
    print("Carregando modelo Word2Vec (pode levar um tempo)...")
    model = KeyedVectors.load_word2vec_format("cbow_s300.txt", binary=False)
    print("Modelo carregado com sucesso.")

    secret = "felicidade" # Mude para testar

    oracle = OfflineContextoOracle(model, secret)
    solver = HybridContextoSolver(model, oracle_query=oracle.query_rank) 
    history = solver.solve(max_attempts=200, verbose=True)

except FileNotFoundError:
    print("\nERRO: Arquivo 'cbow_s300.txt' não encontrado.")
    print("Por favor, baixe o modelo NILC (cbow_s300.txt) e coloque-o na mesma pasta do script.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")