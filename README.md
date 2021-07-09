# Multilabel Document Categorization

This Repository consists of work done for performing multilabel document categorization using both unsupervised and supervised learning.
Please Refer to this [Task Report Document](https://github.com/harshgeek4coder/Multilabel_Document_Categorization_/blob/main/Task%20Report.pdf) For Full Analysis of this Entire Task.

This Task consists of two subtasks :
- Subtask I  : Unsupervised topic modelling
- Subtask II : Learning a supervised multi-Topic Classifier

### Running :
- Clone the Repo
- Activate your virtualenv.
- Run the following script in CLI :
```
pip install -r requirements.txt in your shell.
python main.py
```
- Please NOTE : 
  - You are supposed to put this file : ``` glove.6B.300d.txt ``` in this ```glove directory``` before running ```main.py```. Please Refer to this [Readme](https://github.com/harshgeek4coder/Multilabel_Document_Categorization_/blob/main/glove/README.md) for further instructions.
   - You would also have to download the following files [If Not Already Downloaded]:
   ```
   - nltk.download('stopwords')
   - nltk.download('punkt')
   - nltk.download('wordnet')
   ```

### Root Folder Structure : 
```
│   data_process.py
│   get_features.py
│   inference.py
│   main.py
│   post_process.py
│   prepare_embed_matrix.py
│   process_supervised_data.py
│   requirements.txt
│   save_n_load_state.py
│   supervised_models.py
│   tokenize_n_padding.py
│   unsupervised_models.py
│   utils.py
│
├───datasets
│       pre_processed_df.csv [This file will be automatically added once you run main.py]
│       sentisum-assessment-dataset.csv
│
└───glove
        glove.6B.300d.txt [After downloading and putting this file in glove directory.]
```
