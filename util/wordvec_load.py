# -*- coding: utf-8 -*-

import numpy as np

def LoadGlove(we_path):
    embeddings_index = {}
    with open(we_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def get_glove_embeddings(embeddings_index, dim, tokenizer):
    hits = 0 
    misses = 0 
    vocab_size = len(tokenizer.word_index)+1

    embedding_matrix = np.zeros((vocab_size, dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    return embedding_matrix