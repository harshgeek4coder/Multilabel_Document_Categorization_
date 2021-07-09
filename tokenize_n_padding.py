from utils import *




def get_tokeniser_paddings(X_train, y_train, X_test, y_test,max_len):

    
    # The maximum number of words to be used. (most frequent)
    vocab_size = 50000

    # Latent tensorflow Embedding
    embedding_dim = 256

    # Max number of words in each complaint.
    max_length = max_len+4

    # Truncate and padding options
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'


    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index

    train_seq = tokenizer.texts_to_sequences(X_train)
    train_padded = pad_sequences(train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    validation_seq = tokenizer.texts_to_sequences(y_train)
    validation_padded = pad_sequences(validation_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)



    #Using One Hot Enocder to Enocde our Multi class Labels  :
    encode = OneHotEncoder()

    training_labels = encode.fit_transform(X_test)
    validation_labels = encode.transform(y_test)

    training_labels = training_labels.toarray()
    validation_labels = validation_labels.toarray()



    print("Summary of all shapes of data input ->  \n")
    print("Train_Padded Shape : ",train_padded.shape)
    print("Validation_Labels Shape : ",validation_labels.shape)
    print("Validation_Padded Shape : ",validation_padded.shape)
    print("Train_Labels Shape : ",training_labels.shape)



    return word_index,encode,tokenizer,train_padded,validation_padded,training_labels,validation_labels
