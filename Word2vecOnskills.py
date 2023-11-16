# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:51:50 2023

@author: vinay
"""

import pandas as pd
from gensim.models import Word2Vec,FastText
import numpy as np 

df = pd.read_csv("skill2vec_50K.csv")
skills_columns = df.columns.tolist()[1:]  
skills_data = df[skills_columns].values.tolist()
preprocessed_skills = [
    [str(skill).lower() for skill in row if str(skill).strip()] for row in skills_data
]

model = Word2Vec(preprocessed_skills, vector_size=100, min_count=5, sg=0)
model3 = FastText(sentences=preprocessed_skills, vector_size=100, window=5, min_count=5, sg=0)

#model3.save("fasttextcbow_model3CBOW.bin")
model3.save("fasttext_modelcbow.bin")
model.save("word2veccbow.bin")
"""
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
model = Word2Vec.load("word2vec_model1.bin")
technologies = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Internet of Things",
    "Blockchain",
    "Robotic Process Automation",
    "Augmented Reality",
    "Virtual Reality",
    "Cybersecurity",
    "Cloud Computing",
    "Natural Language Processing",
    "Computer Vision",
    "Big Data Analytics",
    
    
    
]
combined_list = [
    "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "Go", "Swift", "TypeScript",
 
    "HTML", "css", "JavaScript", "React", "Angular", "Flask", "Django", "Flask", "Pyramid", "Bottle"
]
similarity_matrix = []
for tech1 in combined_list:
    row = []
    for tech2 in combined_list:
        similarity = model.wv.similarity(tech1.lower(), tech2.lower())
        row.append(similarity)
    similarity_matrix.append(row)
similarity_matrix = np.zeros((len(combined_list), len(combined_list)))
for i, tech1 in enumerate(combined_list):
    for j, tech2 in enumerate(combined_list):
        similarity = model.wv.similarity(tech1.lower(), tech2.lower())
        similarity_matrix[i, j] = similarity
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(similarity_matrix, cmap="hot")

ax.set_xticks(np.arange(len(combined_list)))
ax.set_yticks(np.arange(len(combined_list)))
ax.set_xticklabels(combined_list, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(combined_list, fontsize=8)


cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)
cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom")
plt.tight_layout()
plt.show()
"""