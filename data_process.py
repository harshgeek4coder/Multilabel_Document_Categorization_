from utils import *


data_path='sentisum-assessment-dataset.csv'

def load_preprocess(data_path):
    
    df=pd.read_csv(data_path,names=['text','text2'],header=0,usecols=['text'])
    documents=df['text'].apply(pre_process)

    return documents