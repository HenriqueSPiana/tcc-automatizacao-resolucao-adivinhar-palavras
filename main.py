import random
import numpy as np
from typing import Callable, List, Tuple, Set
import nltk
from sklearn.linear_model import Ridge
from gensim.models import KeyedVectors
from simplemma import lemmatize
import requests 
import concurrent.futures

import requests

class ContextoAPI:
    BASE_URL = "https://api.contexto.me/machado/pt-br/game"

    def __init__(self, dia: int):
        if not isinstance(dia, int) or dia <= 0:
            raise ValueError("O dia deve ser um número inteiro positivo.")
        self.dia = dia
        print(f"✔ Configurado para resolver o Contexto do dia: {self.dia}")

    def query(self, guess: str) -> int | None:
        """
        Envia 'guess' para a API e retorna o rank (distância).
        Se a palavra for inválida ou ocorrer um erro, retorna None.
        """
        url = f"{self.BASE_URL}/{self.dia}/{guess.lower()}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('distance')
            return None
        except requests.RequestException as e:
            print(f"❌ Erro de rede ao tentar chutar '{guess}': {e}")
            return None

# A CLASSE ContextoSolver PERMANECE EXATAMENTE A MESMA QUE VOCÊ ENVIOU
# Imports necessários: random, numpy, typing, nltk, sklearn, gensim, simplemma
# ...

class ContextoSolver:
    def __init__(
        self,
        model: KeyedVectors,
        query_function: Callable,
        optimization_rank_limit: int = 100,
        max_vocab: int = 150_000,
        random_state: int = 42,
    ):
        self.model = model
        self.query_function = query_function
        self.optimization_rank_limit = optimization_rank_limit
        self.rng = random.Random(random_state)
        
        with open("dicionario_ptbr.txt", "r", encoding="utf-8") as f:
            self.portuguese_dictionary = {line.strip().lower() for line in f}
        self.stopwords = set(nltk.corpus.stopwords.words("portuguese"))
        
        print("Filtrando vocabulário...")
        base_vocab = model.index_to_key[:max_vocab]
        self.vocab = {w for w in base_vocab if self._is_valid_word(w)}
        print(f"Vocabulário filtrado para {len(self.vocab)} palavras válidas.")

        # ### MUDANÇA: Garante que as sementes também sejam lemmas
        default_seeds = ["animal", "objeto", "lugar", "corpo", "natureza", "alimento", "ferramenta", "sentimento"]
        self.seed_words = [self._apply_lemmatizer(w) for w in default_seeds if w in self.vocab]

        self.guessed = set() # Armazena os lemmas que já foram tentados
        self.history = list()
        self.best_words = list()
        self.best_rank = float("inf")

        self.reg = Ridge(alpha=0.5, random_state=random_state)
        self.X_obs = list()
        self.y_obs = list()
        self._fitted = False 
        self.optimization_mode = False

    def _get_batch_candidates(self, batch_size: int) -> List[str]:
        """
        Gera um lote de candidatos de forma otimizada, fazendo o trabalho
        computacional pesado uma única vez.
        """
        # Fase inicial de exploração com palavras-semente
        if len(self.history) < 5:
            seeds = [s for s in self.seed_words if self._is_new(s)]
            return seeds[:batch_size]

        # 1. Gera um grande pool de candidatos brutos UMA VEZ
        raw_candidates = self._get_candidates_from_history()
        
        # 2. Converte para lemmas únicos e válidos
        lemma_candidates = {self._apply_lemmatizer(c) for c in raw_candidates}
        valid_lemmas = [
            lemma for lemma in lemma_candidates 
            if lemma in self.vocab and self._is_new(lemma)
        ]

        if not valid_lemmas:
            # Se não houver candidatos, recorre a uma seleção aleatória
            unguessed = [w for w in list(self.vocab) if self._is_new(w)]
            return self.rng.sample(unguessed, min(len(unguessed), batch_size))

        # 3. Seleciona os melhores 'batch_size' candidatos do pool
        if self._fitted:
            # Modo otimizado: usa a regressão para pontuar todos de uma vez
            cand_vecs = np.vstack([self.model.get_vector(w, norm=True) for w in valid_lemmas])
            preds = self.reg.predict(cand_vecs)
            # Pega os índices dos 'batch_size' melhores candidatos
            best_indices = np.argsort(preds)[::-1][:batch_size]
            return [valid_lemmas[i] for i in best_indices]
        else:
            # Modo de similaridade: calcula a similaridade de todos de uma vez
            top_3_vecs = [self.model.get_vector(w, norm=True) for w, r in self.best_words[:3]]
            scores = []
            for cand in valid_lemmas:
                cand_vec = self.model.get_vector(cand, norm=True)
                avg_sim = sum(np.dot(cand_vec, top_vec) for top_vec in top_3_vecs) / len(top_3_vecs)
                scores.append((avg_sim, cand))
            
            scores.sort(key=lambda x: x[0], reverse=True)
            return [cand for score, cand in scores[:batch_size]]
        

    def _observe_and_learn(self, word: str, rank: int):
        # ### MUDANÇA: A função não precisa mais do lemma do servidor
        self.history.append((word, rank))
        # Adiciona o lemma da palavra que acabamos de tentar
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

    def _get_candidates_from_history(self) -> Set[str]:
        # Este método continua gerando palavras candidatas, que serão lematizadas depois
        # Nenhuma mudança aqui
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
        # Fase inicial de exploração com palavras-semente
        if len(self.history) < 5:
            for seed in self.seed_words:
                if self._is_new(seed):
                    return seed
        
        # Gera candidatos brutos a partir do histórico
        raw_candidates = self._get_candidates_from_history()
        
        # Converte todos os candidatos em seus lemmas únicos
        lemma_candidates = {self._apply_lemmatizer(c) for c in raw_candidates}
        
        # Filtra para manter apenas lemmas válidos (no vocabulário) e que ainda não foram tentados
        valid_lemmas = [
            lemma for lemma in lemma_candidates 
            if lemma in self.vocab and self._is_new(lemma)
        ]

        # Se tivermos lemmas válidos e o modelo de regressão estiver treinado, usa a predição
        if self._fitted and valid_lemmas:
            # Pega o vetor para cada lemma válido
            cand_vecs = np.vstack([self.model.get_vector(w, norm=True) for w in valid_lemmas])
            preds = self.reg.predict(cand_vecs)
            # Retorna o lemma com a maior predição de proximidade
            return valid_lemmas[np.argmax(preds)]

        # Se não, mas ainda tivermos lemmas, escolhe o mais similar às melhores palavras
        if valid_lemmas:
            top_3_vecs = [self.model.get_vector(w, norm=True) for w, r in self.best_words[:3]]
            best_cand, max_sim = None, -2.0
            for cand in valid_lemmas:
                cand_vec = self.model.get_vector(cand, norm=True)
                avg_sim = sum(np.dot(cand_vec, top_vec) for top_vec in top_3_vecs) / len(top_3_vecs)
                if avg_sim > max_sim:
                    max_sim, best_cand = avg_sim, cand
            return best_cand
        
        # Como último recurso, escolhe uma palavra aleatória que ainda não foi tentada
        # Garante que mesmo a escolha aleatória seja um lemma novo
        unguessed_words = [w for w in list(self.vocab) if self._is_new(w)]
        if unguessed_words:
            return self.rng.choice(unguessed_words)
        
        # Se não houver mais nada a tentar (raro)
        raise StopIteration("Não há mais palavras novas para tentar no vocabulário.")

    def solve(self, max_attempts: int = 200, verbose: bool = True, batch_size: int = 8):
        print(f"Iniciando solver com lotes paralelos de tamanho {batch_size}.")
        attempt_num = 1

        while attempt_num <= max_attempts:
            # --- FASE 1: Obter o lote de candidatos de forma otimizada ---
            # O trabalho pesado agora é feito aqui, UMA VEZ por lote.
            batch_to_guess = self._get_batch_candidates(batch_size)
            
            if not batch_to_guess:
                print("Não há mais palavras novas para tentar. Encerrando.")
                break

            # --- FASE 2: Executar o lote em paralelo (sem alterações aqui) ---
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_word = {executor.submit(self.query_function, word): word for word in batch_to_guess}

                for future in concurrent.futures.as_completed(future_to_word):
                    word = future_to_word[future]
                    try:
                        rank = future.result()
                        if rank is not None:
                            results.append({'word': word, 'rank': rank})
                        else:
                            self.guessed.add(self._apply_lemmatizer(word))
                    except Exception as exc:
                        print(f"'{word}' gerou uma exceção: {exc}")

            # --- FASE 3: Aprender com os resultados (sem alterações aqui) ---
            if not results:
                attempt_num += len(batch_to_guess)
                continue

            results.sort(key=lambda x: x['rank'])
            
            found_in_batch = False
            for result in results:
                word, rank = result['word'], result['rank']
                
                # Apenas processa se a palavra (ou seu lemma) não foi registrada por outra thread
                if self._is_new(word):
                    self._observe_and_learn(word, rank)
                else:
                    # Se outra thread já processou uma palavra com o mesmo lemma, ignora
                    continue

                if verbose:
                    mode = "OPTIMIZE" if self.optimization_mode else "EXPLORE"
                    print(f"Tentativa {attempt_num:03d} [{mode}]: {word:<15} → rank {rank:<5} (Melhor: {self.best_rank})")
                
                attempt_num += 1

                if rank == 1:
                    print(f"\n🎉 Descoberta em {attempt_num - 1} tentativas: '{word}'")
                    found_in_batch = True
                    break
            
            if found_in_batch:
                break

        if self.best_rank != 1 and self.best_words:
            print(f"\nNão encontrou a palavra. Melhor tentativa: '{self.best_words[0][0]}' (rank {self.best_words[0][1]})")

        return self.history


    def _is_valid_word(self, w: str) -> bool:
        return (w.isalpha() and w.islower() and w not in self.stopwords and len(w) > 2 and w in self.portuguese_dictionary)
    
    def _apply_lemmatizer(self, w: str) -> str:
        # A função de lematização agora é o centro da nossa estratégia
        return lemmatize(w, lang="pt")
    
    def _is_new(self, w: str) -> bool:
        """Verifica se o lemma da palavra 'w' já foi tentado."""
        return self._apply_lemmatizer(w) not in self.guessed


if __name__ == "__main__":
    nltk.download('stopwords', quiet=True)
    nltk.download('rslp', quiet=True)

    print("Carregando modelo Word2Vec (pode levar um tempo)...")
    # Certifique-se de que o arquivo cbow_s300.txt está no mesmo diretório
    model = KeyedVectors.load_word2vec_format("cbow_s300.txt", binary=False)
    print("Modelo carregado com sucesso.")

    # --- CONFIGURAÇÃO DA EXECUÇÃO ---
    # Substitua pelo número do dia que você quer resolver.
    # Você pode encontrar o número do dia atual na URL do site Contexto.
    DIA_DO_JOGO = 971
    contexto = ContextoAPI(dia=DIA_DO_JOGO)
    solver = ContextoSolver(model, query_function=contexto.query)
    
    # Inicia a resolução, tentando 8 palavras de cada vez em paralelo
    history = solver.solve(max_attempts=200, verbose=True, batch_size=8)

    print("\n--- Fim da execução ---")
