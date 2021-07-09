from utils import *
from post_process import get_post_processed_df
from data_process import pre_process_df



def get_x_y(text,labels):
    X_train, y_train, X_test, y_test = train_test_split(text,labels, test_size = 0.20, random_state = 2, stratify=labels)
    print(X_train.shape,X_test.shape)
    print(y_train.shape,y_test.shape)

    #Reshaping Labels Input :
    X_test=X_test.reshape(-1,1)
    y_test=y_test.reshape(-1,1)

    print(X_train.shape,X_test.shape)
    print(y_train.shape,y_test.shape)

    max_len=get_max_seq_len(text)

    return X_train, y_train, X_test, y_test,max_len