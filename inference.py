from utils import *

topic_names=['value for money','garage service', 'mobile fitter', 
             'change of date', 'wait time', 'delivery punctuality',
             'ease of booking' , 'location', 'booking confusion' , 'tyre quality' ,
             'length of fitting' ,  'discounts']



def get_inference_from_supervised(text,model,max_len,tokenizer,encode):  
  
  print("Inference and Results from Supervised model Inference : \n")
  new_text = [clean_text(text)]
  print(text)
  print(new_text)
  seq = tokenizer.texts_to_sequences(new_text)
  padded = pad_sequences(seq, maxlen=max_len, padding=padding_type, truncating=trunc_type)
  pred = model.predict(padded)
  acc = model.predict_proba(padded)
  a=acc[0]
  idx=heapq.nlargest(3,range(len(a)),a.take)
  n_cat=topic_names[idx[0]],topic_names[idx[1]],topic_names[idx[2]]
  predicted_label = encode.inverse_transform(pred)
  print('')
 
  print(f'Predicted Topic is: {predicted_label[0]}')
  print("Top 3 Topic Categories : ", n_cat)

  return n_cat


def get_inference_from_unsupervised_model(model, vectorizer, text, threshold):
    v_text = vectorizer.transform([text])
    score = model.transform(v_text)

    labels = set()
    
    for i in range(len(score[0])):
        if score[0][i] > threshold:
            labels.add(topic_names[i])
    
    a=np.array(score[0])
    idx=heapq.nlargest(3,range(len(a)),a.take)
    n_cat=topic_names[idx[0]],topic_names[idx[1]],topic_names[idx[2]]
    
    if not labels:
        return 'None', -1, set()

    print("Inference and Results from Unsupervised model Inference : \n")
    return topic_names[np.argmax(score)], score, labels,idx,n_cat
