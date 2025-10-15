import random
import numpy as np
from typing import Callable
import nltk
from sklearn.linear_model import Ridge
from gensim.models import KeyedVectors
import nltk
from contexto_offline import ContextoOffline
from contexto_online import ContextoOnline
from contexto_api import ContextoAPI
from simplemma import lemmatize
from prepare_embedding import load_huggingface_to_gensim

class ContextoSolver:
    def __init__(
        self,
        model: KeyedVectors,
        query_function: Callable,
        optimization_rank_limit: int = 100, # quando atingir esse rank, ativa modo otimização
        max_vocab: int = 150_000, # limita o vocabulário para acelerar o processamento e evitar palavras estranhas
        random_state: int = 42, # serve para ter resultado reproduzível
    ):
        self.model = model
        self.query_function = query_function
        self.optimization_rank_limit = optimization_rank_limit
        self.rng = random.Random(random_state)
        
        with open("dicionario_ptbr.txt", "r", encoding="utf-8") as f:
            self.portuguese_dictionary = {line.strip().lower() for line in f}
        self.stopwords = set(nltk.corpus.stopwords.words("portuguese"))
        self.stemmer = nltk.stem.RSLPStemmer()

        print("Filtrando vocabulário com base no dicionário (pode levar um momento)...")
        base_vocab = model.index_to_key[:max_vocab]
        self.vocab = {w for w in base_vocab if self._is_valid_word(w)}
        print(f"Vocabulário filtrado para {len(self.vocab)} palavras válidas.")

        default_seeds = ["animal", "objeto", "lugar", "corpo", "natureza", "alimento", "ferramenta", "sentimento"]
        self.seed_words = [w for w in default_seeds if w in self.vocab]

        self.guessed = set()
        self.history = list()
        self.best_words = list()
        self.best_rank = float("inf")

        self.reg = Ridge(alpha=0.5, random_state=random_state)
        self.X_obs = list()
        self.y_obs = list()
        self._fitted = False 
        self.optimization_mode = False

    def _observe_and_learn(self, word: str, rank: int):
        self.history.append((word, rank))
        self.guessed.add(self._apply_lemmatizer(word))
        self.best_words.append((word, rank))
        self.best_words.sort(key=lambda x: x[1])
        
        if rank < self.best_rank:
            self.best_rank = rank
            if not self.optimization_mode and self.best_rank < self.optimization_rank_limit:
                self.optimization_mode = True
                print(f"\n--- ATIVANDO MODO DE OTIMIZAÇÃO (MELHOR RANK: {self.best_rank}) ---\n")

        if self.optimization_mode:
            y = 1.0 / np.log1p(rank)
            self.X_obs.append(self.model.get_vector(word, norm=True))
            self.y_obs.append(y)

            if len(self.X_obs) >= 15:
                self.reg.fit(np.vstack(self.X_obs), np.array(self.y_obs))
                self._fitted = True

    def _get_candidates_from_history(self) -> set:
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
                if self._is_new(seed):
                    return seed
            return self.rng.choice([w for w in list(self.vocab) if self._is_new(w)])

        candidates = self._get_candidates_from_history()
        valid_candidates = [c for c in candidates if self._is_new(c)]

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
        
        return self.rng.choice([w for w in list(self.vocab) if self._is_new(w)])

    def solve(self, max_attempts: int = 200, verbose: bool = True) -> list:
        for attempt in range(1, max_attempts + 1):
            current = self._choose_next()
            if current in self.guessed: continue

            rank = self.query_function(current)
            if rank is None:
                self.guessed.add(self._apply_lemmatizer(current))
                continue
            
            self._observe_and_learn(current, rank)
            
            if verbose:
                mode = "OPTIMIZE" if self.optimization_mode else "EXPLORE"
                print(f"Tentativa {attempt:03d} [{mode}]: {current:<15} → rank {rank:<5} (Melhor: {self.best_rank})")

            if rank == 1:
                print(f"Descoberta em {attempt} tentativas: {current}")
                break
        
        if self.best_rank != 1:
            print(f"Não encontrou a palavra. Melhor tentativa: '{self.best_words[0][0]}' (rank {self.best_words[0][1]})")

        return self.history
    
    def _is_valid_word(self, w: str) -> bool:
        """
        Verifica se a palavra é válida:
        1. Apenas letras.
        2. Minúscula.
        3. Não é uma stopword.
        4. Tem mais de 2 letras.
        5. Existe no dicionário pt-br.
        """
        return (
            w.isalpha() and
            w.islower() and
            w not in self.stopwords and
            len(w) > 2 and
            w in self.portuguese_dictionary
        )
    
    def _apply_stemmer(self, w: str) -> str:
        """Aplica a stemização na palavra."""
        return self.stemmer.stem(w)
    
    def _apply_lemmatizer(self, w: str) -> str:
        """Aplica a lematização na palavra."""
        return lemmatize(w, lang="pt")
    
    def _is_new(self, w: str) -> bool:
        return self._apply_lemmatizer(w) not in self.guessed

if __name__ == "__main__":
    nltk.download('stopwords', quiet=True)
    nltk.download('rslp', quiet=True)

    print("Carregando modelo (pode levar um tempo)...")
    # model = KeyedVectors.load_word2vec_format("cbow_s300.txt", binary=False)
    # model = load_huggingface_to_gensim("nilc-nlp/fasttext-skip-gram-300d")
    # model = load_huggingface_to_gensim("nilc-nlp/word2vec-skip-gram-300d")
    model = load_huggingface_to_gensim("nilc-nlp/glove-300d")
    print("Modelo carregado com sucesso.")

    # secret = "apartamento"
    # contexto = ContextoOffline(model, secret)
    # solver = ContextoSolver(model, query_function=contexto.query_rank) 
    # history = solver.solve(max_attempts=200, verbose=True)
    
    # contexto = ContextoOnline(headless=True).start()
    # solver = ContextoSolver(model, query_function=contexto.query)
    # history = solver.solve(max_attempts=200, verbose=True)
    # contexto.close()
    
    contexto = ContextoAPI(dia=1000)
    solver = ContextoSolver(model, query_function=contexto.query)
    history = solver.solve(max_attempts=100, verbose=True)
