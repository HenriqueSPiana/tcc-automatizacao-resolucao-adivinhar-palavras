import numpy as np
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

def load_huggingface_to_gensim(repo_id: str) -> KeyedVectors:
    path = hf_hub_download(repo_id, "embeddings.safetensors")

    vectors = load_file(path)["embeddings"]
    
    vocab_path = hf_hub_download(repo_id, "vocab.txt")
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [w.strip() for w in f]

    kv = KeyedVectors(vector_size=vectors.shape[1])
    kv.add_vectors(vocab, np.array(vectors, dtype=np.float32))
    return kv