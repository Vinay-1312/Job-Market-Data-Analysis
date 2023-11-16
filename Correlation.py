# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:45:04 2023

@author: vinay
"""

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, FastText

# Load the pretrained Word2Vec or FastText model
# Replace "model_path" with the actual path to your pretrained model file
#word2vec2_model = Word2Vec.load('word2vec_model1.bin')
word2vec2_model = Word2Vec.load("word2veccbow_model.bin")
#word2vec2_model = Word2Vec.load('word2veccbow_model1.bin')
# Load the trained FastText model
#fasttext_model = FastText.load('word2veccbow_model.bin')
fasttext_model = FastText.load('fasttextcbow_model.bin')
#fasttext_model = FastText.load('fasttextcbow_modelResume.bin')
#word2vec2_model = Word2Vec.load('word2veccbow_model.bin')
# List of programming languages
combined_list = [
    "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "Golang", "Swift", "Kotlin", "TypeScript",
 
    "HTML", "css", "JavaScript", "React", "Angular", "Django", "Flask"]

it_skills = [
    "SQL", "HTML", "CSS", "Git", "Linux", "AWS", "Azure",'mongo db','windows',
    "Docker", "Kubernetes", "Big Data", "Machine Learning", "Data Analysis", "Cybersecurity",
    "DevOps", "Agile", "Network Security", "Virtualization", "IT Project Management",
    "IT Service Management", "Database Administration"
]

soft_skills = [
    "Communication skills",
    "Teamwork and collaboration",
    "Problem-solving",
    "Adaptability and flexibility",
    "Time management",
    "Leadership",
    "Emotional intelligence",
    "Critical thinking",
    "Creativity",
    "Interpersonal skills",
    "Conflict resolution",
    "Decision-making",
    "Active listening",
    "Empathy"
]
latest_technologies = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Internet of Things",
    "Edge Computing",
    "Blockchain",
    "Robotic Process Automation",
    "Augmented Reality",
    "Virtual Reality",
    "5G Technology",
    "Quantum Computing",
    "Cybersecurity",
    "Cloud Computing",
    "Serverless Architecture",
    "Natural Language Processing",
    "Computer Vision",
    "Big Data Analytics",
    
]


# Load the trained Word2Vec model


# Input job title

similarity_matrix = []
#similarity_matrix = np.zeros((len(programming_languages), len(programming_languages)))
def getCorrelation(data):
    def get_similarity(skill1, skill2):
        #if skill1 in word2vec_model.wv.key_to_index and skill2 in word2vec_model.wv.key_to_index:
            #return word2vec_model.wv.similarity(skill1, skill2)
        #elif# skill1 in fasttext_model.wv.key_to_index and skill2 in fasttext_model.wv.key_to_index:
        if  skill1 in word2vec2_model.wv.key_to_index and skill2 in word2vec2_model.wv.key_to_index: 
            return word2vec2_model.wv.similarity(skill1, skill2)
        else:
               return fasttext_model.wv.similarity(skill1, skill2)
    for i in range(len(data)):
        row = []
        for j in range(len(data)):
            if i == j:
                row.append(1.0)  # Skills are identical, so similarity is 1
            else:
                similarity = get_similarity(data[i].lower(), data[j].lower())
                if similarity is not None:
                    row.append(similarity)
                else:
                    row.append(0.0)  # Skills not found in any model, so similarity is 0
        print(row)
        similarity_matrix.append(row)
            
        
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity_matrix, cmap="hot")
    
    # Set x-axis and y-axis labels
    ax.set_xticks(np.arange(len(data)))
    ax.set_yticks(np.arange(len(data)))
    ax.set_xticklabels(data, rotation=45, ha="center", fontsize=8)
    ax.set_yticklabels(data, fontsize=8)
    
    # Set the colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)
    cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

getCorrelation(combined_list)
getCorrelation(it_skills)
getCorrelation(soft_skills)
getCorrelation(latest_technologies)



