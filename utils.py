import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation,TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from nltk.tokenize import word_tokenize
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


STOPWORDS = set(stopwords.words('english'))
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional,Conv1D,MaxPool1D,MaxPooling1D,GlobalMaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import time
import heapq


import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


import os
os.mkdir('.\saved_models_state')


stemmer=PorterStemmer()
wnl = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))
remove_brackets = re.compile('[/(){}\[\]\|@,;]')
bad_symbols = re.compile('[^0-9a-z #+_]')
remove_num = re.compile('[\d+]')
#Do input the words in the external_Stop_words_list , if you wish to remove those words in your text corpus.
external_stop_words_list=['redacted']

def clean_text(text,stop_words=False,stemming=False,lemma=False,expand_terms=False,external_stop=False):
    
    """
    input: a text string
    return: a pre-processed clean string
    """
    text = text.lower() 
    if expand_terms==True:  
        text = re.sub(r"it\'s","it is",str(text))
        text = re.sub(r"i\'d","i would",str(text))
        text = re.sub(r"don\'t","do not",str(text))
        text = re.sub(r"ain\'t","are not",str(text))
        text = re.sub(r"aren\'t","are not",str(text))
        text = re.sub(r"he\'s","he is",str(text)) 
        text = re.sub(r"there\'s","there is",str(text)) 
        text = re.sub(r"that\'s","that is",str(text)) 
        text = re.sub(r"can\'t", "can not", text) 
        text = re.sub(r"cannot", "can not ", text) 
        text = re.sub(r"what\'s", "what is", text) 
        text = re.sub(r"What\'s", "what is", text) 
        text = re.sub(r"\'ve ", " have ", text) 
        text = re.sub(r"n\'t", " not ", text) 
        text = re.sub(r"i\'m", "i am ", text) 
        text = re.sub(r"I\'m", "i am ", text) 
        text = re.sub(r"\'re", " are ", text) 
        text = re.sub(r"\'d", " would ", text) 
        text = re.sub(r"\'ll", " will ", text) 
        text = re.sub(r"\'s"," is",text)

    else:
        pass

    text = re.sub(r'https?://[^\s]+', '', text)
        
    text = remove_brackets.sub(' ', text)  

    text = remove_num.sub('', text)

    text = bad_symbols.sub('', text)
  
    if stop_words==True:
        text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
    else:
        pass

    if stemming==True:
        text = ' '.join([stemmer.stem(word) for word in text.split()])
    else:
        pass

    if lemma==True:
        text = ' '.join([wnl.lemmatize(word) for word in text.split()])
    else:
        pass 

    if external_stop==True:
        
        text = ' '.join(word for word in text.split() if word not in external_stop_words_list)
    else:
        pass


    return text

def pre_process(text):
    text=clean_text(str(text),stop_words=True,lemma=True,external_stop=True)
           
    return text    



def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

    return None

def get_max_seq_len(corpus):
  corpus_len=[]

  for question in corpus:
    question_list=(str(question)).split()
    corpus_len.append(len(question_list))

  return max(corpus_len)


def plot_graphs(history):
  plt.title('Loss VS Accuracy')
  plt.plot(history.history['loss'], label='train loss')
  plt.plot(history.history['val_loss'], label='test loss')
  plt.legend()
  
  plt.plot(history.history['accuracy'], label='train accuracy')
  plt.plot(history.history['val_accuracy'], label='test accuracy')
  plt.legend()
  plt.show()
  return None



# The maximum number of words to be used. (most frequent)
vocab_size = 50000

# Latent tensorflow Embedding
embedding_dim = 256

# Truncate and padding options
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

print("Imported All Relevant Packages and Helper Functions Successfully..!")
