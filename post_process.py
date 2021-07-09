from utils import *
from unsupervised_models import *
from get_features import *
from data_process import *


data_path='sentisum-assessment-dataset.csv'
documents=load_preprocess(data_path)
df=pd.read_csv(data_path,names=['text','text2'],header=0,usecols=['text'])


tfidf_vectorizer,tfidf_mat,tfidf_feature_names=get_tfidf_features(documents)
tf_vectorizer,tf_mat,tf_feature_names=get_countvec_features(documents)

nmf_model=get_nmf_model(tfidf_mat)
svd_model=get_svd_model(tfidf_mat)
lda_model=get_lda_model(tf_mat)


def get_post_processed_df():
    output=lda_model.transform(tf_vectorizer.fit_transform(documents))
    topic_names = ["Topic" + str(i) for i in range(lda_model.n_components)]
    document_names = ["Doc" + str(i) for i in range(len(documents))]
    df_final = pd.DataFrame(np.round(output, 2), columns=topic_names, index=document_names)

    main_topic = np.argmax(df_final.values, axis=1)
    df_final['main_topic'] = main_topic

   
    df_final['Text']=df['text'].to_numpy()


    #Final Order Topic Wise :

    topics_dict_2={
    0:'value for money',1:'garage service',2: 'mobile fitter',
             3:'change of date',4: 'wait time', 5:'delivery punctuality', 6:'ease of booking' ,
             7:'location', 8:'booking confusion' ,9: 'tyre quality' ,10: 'length of fitting',  11:'discounts' }
    


    new_df=df_final
    new_df['target_label']='temp'
    new_df['target_label']=new_df['main_topic'].map(topics_dict_2)

    print("Exporting Fully post processed dataframe now..!")
    new_df.to_csv('pre_processed_df.csv')
    print("Exported..")

    return new_df



"""
This will return the FULLY PRE PROCESSED DATAFRAME For sub task 2

"""