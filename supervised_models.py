from utils import *


def create_model(train_padded,validation_padded):
    print("Now Preparing Model..")

    tf.keras.backend.clear_session()
    model = Sequential()

    model.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
    model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(embedding_dim)))
    model.add(Dropout(0.3))
    model.add(Dense(12, activation='softmax'))

    model.summary()
    optim = tf.keras.optimizers.Adam()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optim,
        metrics=['accuracy'],
    )

    print("Model Built..")
    return model


def train_model(model,train_padded,validation_padded,training_labels,validation_labels):
    epochs = 10
    batch_size = 32
    
    tqdm_callback = tfa.callbacks.TQDMProgressBar()
    
    callbacks_list = [EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=0),tqdm_callback]

    
    print("Now Training Model..")
    history=model.fit(train_padded, training_labels, shuffle=True ,
                        epochs=epochs, batch_size=batch_size, verbose=0,
                        callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))

    print("Model Trained..")
    print("Now evaluating model .. \n")
    loss,acc=model.evaluate(validation_padded, validation_labels)
    print("Validation Loss : ",loss)
    print("Validation Accuracy : ",acc)

    

    return history,model



def create_model_with_embeddings(train_padded,validation_padded,embedding_matrix):
    print("Now Preparing Model with embeddings matrix..")
    embedding_dim=300
    tf.keras.backend.clear_session()
    model = Sequential()

    model.add(Embedding(vocab_size,output_dim=embedding_dim,weights=[embedding_matrix],input_length=train_padded.shape[1]))
    model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(embedding_dim)))
    model.add(Dropout(0.3))
    model.add(Dense(12, activation='softmax'))

    model.summary()
    optim = tf.keras.optimizers.Adam()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optim,
        metrics=['accuracy'],
    )

    print("Model Built with embeddings matrix..")
    return model
