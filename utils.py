import numpy as np
import pandas as pd

import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import NMF, LatentDirichletAllocation,TruncatedSVD
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


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

print("Imported All Relevant Packages and Helper Functions Successfully..!")
