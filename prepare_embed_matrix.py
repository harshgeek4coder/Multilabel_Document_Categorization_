
from utils import *

def prepare_embeddings(word_index):    

    print("Now Preparing Embedding Matrix...!")

    #word_index = tokenizer.word_index
    glove_dir = ".\glove\glove.6B.300d.txt"
    embeddings_index = {}


    f = open(glove_dir,encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    print(word_index)


    embedding_dim = 300
    max_words = 50000              

    embedding_matrix = np.zeros((max_words, embedding_dim))

    for word, i in word_index.items():

        if i < max_words:

            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                
                embedding_matrix[i] = embedding_vector

    print("Embedding Matrix Created...!")
    return embedding_matrix