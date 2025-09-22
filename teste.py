# import pandas as pd
# import spacy
# import nltk

# df_palavras = pd.DataFrame(['amigos', 'amigas', 'amizade', 'carreira', 'carreiras'], columns=['Original'])
# df_palavras

# nltk.download('rslp')

# stemmer = nltk.stem.RSLPStemmer()

# df_palavras['nltk_stemmer'] = [stemmer.stem(palavra) for palavra in df_palavras['Original']]

# nlp = spacy.load('pt')

# doc = nlp(str([palavra for palavra in df_palavras['Original']]))

# df_palavras['spacy_lemma'] = [token.lemma_ for token in doc if token.pos_ == 'NOUN']

# print(df_palavras)

import pandas as pd
import spacy
import nltk

df_palavras = pd.DataFrame(
    ['apresentar', 'apresentou', 'apresentassem', 'apresentariam',
     'aconteçam', 'surgiu', 'surgir', 'surgirá', 'fazer', 'fazei'],
    columns=['Original']
)

nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()
df_palavras['nltk_stemmer'] = [stemmer.stem(p) for p in df_palavras['Original']]

nlp = spacy.load('pt_core_news_lg')

# ✅ Um Doc por palavra (mesmo número de itens do DataFrame)
docs = list(nlp.pipe(df_palavras['Original'].astype(str)))

df_palavras['spacy_pos']   = [d[0].pos_   if len(d) else None for d in docs]
df_palavras['spacy_lemma'] = [d[0].lemma_ if len(d) else None for d in docs]
df_palavras['spacy_lemma_noun'] = [
    d[0].lemma_ if len(d) and d[0].pos_ == 'NOUN' else None
    for d in docs
]

print(df_palavras)
