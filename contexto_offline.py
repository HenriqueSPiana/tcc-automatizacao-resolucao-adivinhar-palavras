from gensim.models import KeyedVectors
import numpy as np

class ContextoOffline:
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

    def query_rank(self, guess: str) -> int | None:
        return self.rank_map.get(guess, None)