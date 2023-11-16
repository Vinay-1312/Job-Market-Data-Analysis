    # -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:18:10 2023

@author: vinay
"""

import sqlite3
from gensim.models import Word2Vec, Phrases,FastText
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import fasttext
import re
#nltk.download('punkt')
with open('skill_list.txt', 'r') as file:
    lines = file.readlines()
    
stop_words = set(stopwords.words('english'))

df = pd.DataFrame({'Text': lines})

data2 = pd.read_csv("resume.csv")
conn = sqlite3.connect('JobsData.db')

query = 'SELECT description FROM jobs'
jd = pd.read_sql_query(query, conn)
data = pd.read_csv("merged_requirements.csv")

combined_text = pd.concat([data['Merged Requirements'], jd['description'],data2['Resume_str']], axis=0, ignore_index=True)

combined_data = pd.concat([data['Merged Requirements'],jd['description'],df['Text']],ignore_index=True)
combined_data_file = 'combined_job_descriptions.txt'
combined_data.to_csv(combined_data_file, index=False, header=False)

#job_descriptions = data['Merged Requirements'].tolist()
#sentences = [nltk.word_tokenize(desc.lower()) for desc in job_descriptions]
#sentences = combined_text.str.split()
preprocessed_data = []
# Create bigrams for better phrase detection
#bigram_transformer = Phrases(sentences)
#sentences_with_bigrams = [bigram_transformer[sentence] for sentence in sentences]
for job_desc in combined_data:
    # Lowercasing and removing non-alphanumeric characters
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', job_desc.lower())
    
    # Tokenization
    tokens = word_tokenize(cleaned_text)
    
    # Removing stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Adding preprocessed job description to the list
    preprocessed_data.append(filtered_tokens)

model = Word2Vec(preprocessed_data,  vector_size=100, window=5, min_count=1, sg=0)
model2 = fasttext.train_unsupervised(combined_data_file)
model3 = FastText(sentences=preprocessed_data, vector_size=100, window=5, min_count=1, sg=0)

model3.save("fasttextcbow.bin")l
similar_words = model.wv.most_similar("python")

# Printing similar words
for word in similar_words:
    print(word)
model.save("word2veccbow.bin")
#model2.save_model("FastText.bin")


"""
from gensim.models import Word2Vec,FastText
import fasttext
model = Word2Vec.load("word2vec_model.bin")
model2= fasttext.load_model("FastText.bin")
skills = ['python', 'java', 'machine learning', 'data analysis','communication',"artificial integlligence",'data_science',"hardworking","data_analysis","API","AI","MachineLearning","dataScience"]

for skill in skills:
    if skill in model.wv.key_to_index:
        similar_skills = model.wv.most_similar(skill)
       
        for similar_skill, similarity in similar_skills:
            print(similar_skill)
            print(similarity)
        print()
    else:
        print(f"'{skill}' is not in the vocabulary of the Word2Vec model.")
        similar_skills2 = model2.get_nearest_neighbors(skill, k=5)

        print(f"Similar skills for '{skill}':")
        for similar_skill, similarity in similar_skills:
            print(f"{similar_skill}: {similarity:.4f}")

"""
"""
import gensim.downloader
import gensim

#glove_vectors1 = gensim.load('fasttext-wiki-news-subwords-300')
from gensim.models import KeyedVectors
from gensim import models

word2vec_path = 'word2vec-google-news-300.gz'
w2v_model = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
word2vec_path = 'fasttext-wiki-news-subwords-300.gz'
w2v_model2 = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
word2vec_path = 'glove-twitter-200.gz'
encoding = 'latin1'
w2v_model3 = models.KeyedVectors.load_word2vec_format(word2vec_path, encoding = 'latin1')
#glove_vectors2 = gensim.load('word2vec-google-news-300')
#glove_vectors3 = gensim.downloader.load('glove-twitter-200')
print(w2v_model2.most_similar('Artificia Intelligence'))
#print(glove_vectors2.most_similar('Machine Learning'))
#print(glove_vectors3.most_similar('Data Analysis'))
"""
