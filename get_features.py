from utils import *
from data_process import load_preprocess


data_path='sentisum-assessment-dataset.csv'
documents=load_preprocess(data_path)

no_features = 12000

#For NMF and SVD : 
def get_tfidf_features(documents):

    tfidf_vectorizer = TfidfVectorizer( min_df=0.02, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    return tfidf_vectorizer,tfidf,tfidf_feature_names

#For LDA
def get_countvec_features(documents):
    tf_vectorizer = CountVectorizer( min_df=0.02, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    return tf_vectorizer,tf,tf_feature_names