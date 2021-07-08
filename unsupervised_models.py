from utils import *
from get_features import *
from  data_process import *


data_path='sentisum-assessment-dataset.csv'
documents=load_preprocess(data_path)

tfidf_vectorizer,tfidf,tfidf_feature_names=get_tfidf_features(documents)
tf_vectorizer,tf,tf_feature_names=get_countvec_features(documents)

no_topics = 12


def get_nmf_model(tfidf):
    nmf_model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd',max_iter=50)
    nmf_model =nmf_model.fit(tfidf)
    return nmf_model


def get_svd_model(tfidf):
    svd_model = TruncatedSVD(n_components=no_topics, algorithm='randomized', n_iter=50, random_state=1)
    svd_model =svd_model.fit(tfidf)
    return svd_model



def get_lda_model(tf):
    lda_model = LatentDirichletAllocation(n_components=no_topics,learning_decay=0.75, max_iter=50,learning_offset=30, learning_method='online',random_state=1)
    lda_model=lda_model.fit(tf)

    return lda_model



def perform_gridsearch():

    search_params = {'n_components': [12], 'learning_decay': [.5, .7, .9,0.75, 0.80, 0.85],'learning_offset':[30,50]}


    lda = LatentDirichletAllocation()

    
    model = GridSearchCV(lda, search_params,cv=3,verbose=1)

    model.fit(tf)
    best_lda_model = model.best_estimator_
    print("Best model's params: ", model.best_params_)
    print("Best log likelihood score: ", model.best_score_)
    print("Model perplexity: ", best_lda_model.perplexity(tf))

    return best_lda_model

"""
Fitting 3 folds for each of 12 candidates, totalling 36 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done  36 out of  36 | elapsed: 10.7min finished
Best model's params:  {'learning_decay': 0.75, 'learning_offset': 30, 'n_components': 12}
Best log likelihood score:  -83160.21471701778
Model perplexity:  55.26047238754284
"""
