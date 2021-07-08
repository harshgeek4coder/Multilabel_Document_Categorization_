from utils import *
from unsupervised_models import *
from get_features import *
from data_process import *
from post_process import *

data_path='sentisum-assessment-dataset.csv'
documents=load_preprocess(data_path)
print("Data loaded..!")

tfidf_vectorizer,tfidf_mat,tfidf_feature_names=get_tfidf_features(documents)
tf_vectorizer,tf_mat,tf_feature_names=get_countvec_features(documents)
print("Features Extracted..!")

no_topics = 12
no_top_words = 15

print("Now Running Models.. ")
nmf_model=get_nmf_model(tfidf_mat)
svd_model=get_svd_model(tfidf_mat)
lda_model=get_lda_model(tf_mat)

print("Extracting Topic Clusters.. ")
topics_nmf=display_topics(nmf_model, tfidf_feature_names, no_top_words)
topics_svd=display_topics(svd_model, tfidf_feature_names, no_top_words)
topics_lda=display_topics(lda_model, tf_feature_names, no_top_words)

print("Topics Obtained : \n")
print(topics_lda)

print("Now building post processed dataframe for supervised task.. ")
df=get_post_processed_df()
print(df.head())