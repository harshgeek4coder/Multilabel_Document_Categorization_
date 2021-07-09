from utils import *
from unsupervised_models import get_lda_model,get_svd_model,get_nmf_model
from get_features import get_tfidf_features,get_countvec_features
from data_process import load_preprocess,pre_process_df
from post_process import get_post_processed_df
from process_supervised_data import get_x_y
from tokenize_n_padding import get_tokeniser_paddings
from supervised_models import create_model,train_model,create_model_with_embeddings
from prepare_embed_matrix import prepare_embeddings
from save_n_load_state import save_model_state,load_model_state
from inference import get_inference_from_supervised,get_inference_from_unsupervised_model


data_path='.\datasets\sentisum-assessment-dataset.csv'
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

df_text=pre_process_df(df)

text = df_text.values
labels = df['target_label'].values


X_train, y_train, X_test, y_test,max_len=get_x_y(text,labels)



word_index,encoder,tokenizer,train_padded,validation_padded,training_labels,validation_labels=get_tokeniser_paddings(X_train, y_train, X_test, y_test,max_len)

embedding_matrix=prepare_embeddings(word_index)


"""
Please Note - > You can select any one of the following - > 
'model' - to train on latent tensorflow embeddings
'model2' - to train model with glove-300-D Embeddings
"""
model=create_model(train_padded,validation_padded)

model2=create_model_with_embeddings(train_padded,validation_padded,embedding_matrix)

history,model=train_model(model2,train_padded,validation_padded,training_labels,validation_labels)

save_model_state(model,tokenizer)

loaded_model, loaded_tokenizer=load_model_state()



"""
Please Note - > You can select any one of the following - > 
'get_inference_from_supervised' - to get inference from supervised classifier
'get_inference_from_unsupervised_model' - to get inference from unsupervised model
"""
sample_text="Delivered what I ordered and had in stock, excellent fitting service, price was decent and on time."
get_inference_from_supervised(sample_text,loaded_model,max_len,loaded_tokenizer,encoder)
get_inference_from_unsupervised_model(lda_model,tf_vectorizer,sample_text,threshold=0)



