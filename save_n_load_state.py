from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import model_from_json


def save_model_state(model,tokenizer):

  print("Saving Models ..")
  # Save the trained weights
  model.save_weights(r'D:\SentiSum Taskk\NLP Exercise Dataset\final\saved_models_state\model_weights.h5')

  # Save the model architecture
  with open(r'D:\SentiSum Taskk\NLP Exercise Dataset\final\saved_models_state\model_architecture.json', 'w') as f:
    f.write(model.to_json())

  # Save the tokenizer
  with open(r'D:\SentiSum Taskk\NLP Exercise Dataset\final\saved_models_state\tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())
       
  print("Models Saved Successfully With Tokenizer!")



def load_model_state():

  print("Loading Models ..")

  with open(r'D:\SentiSum Taskk\NLP Exercise Dataset\final\saved_models_state\tokenizer.json',encoding="utf8") as f:
    tokenizer = tokenizer_from_json(f.read())

    # Model reconstruction from JSON file
  with open(r'D:\SentiSum Taskk\NLP Exercise Dataset\final\saved_models_state\model_architecture.json', 'r',encoding="utf8") as f:
    model = model_from_json(f.read())

     # Load weights into the new model
  model.load_weights(r'D:\SentiSum Taskk\NLP Exercise Dataset\final\saved_models_state\model_weights.h5')
    
  print("Loaded Models and Tokenizer Successfully !")

  return model, tokenizer


 