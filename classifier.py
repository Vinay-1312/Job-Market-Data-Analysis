# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 07:39:25 2023

@author: vinay
"""

import nltk
import string
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import wordtokenize,senttokenize
from nltk.corpus import stopwords
import gensim
import numpy as np
from gensim.models import Word2Vec,FastText
from sklearn.modelselection import traintestsplit
from sklearn.modelselection import GridSearchCV,RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naivebayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracyscore, precisionscore, recallscore, f1score, rocaucscore
from sklearn.metrics import confusionmatrix
from tensorflow.keras.preprocessing.sequence import padsequences
from sklearn.neuralnetwork import MLPClassifier
def plotconfusionmatrix(cm, classes,title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tightlayout()
#sns.countplot(x=data["Label"])
data = pd.readcsv("data.csv")
#droppping missing values
data.dropna(inplace=True)
lemmatizer = WordNetLemmatizer()
processedText = []

for index, row in data.iterrows():
    print(row[0])
    print(row['Text'])
    text = row['Text']
    tokens = wordtokenize(text)
    tokens= [t.lower() for t in tokens]
    stopwords = set(stopwords.words('english'))

  
    filteredtokens = [token for token in tokens if token.lower() not in stopwords]
    #lemmatizedtokens = [lemmatizer.lemmatize(token) for token in filteredtokens]
    #embeddings = [Word2Vec(word)  for word in lemmatizedtokens]
    #wordembeddings.append(embeddings)
    processedText.append(filteredtokens)



maxseqlength = max(len(text) for text in processedText) 
fastmodel = FastText(processedText,mincount=1,vectorsize=30)
word2vecmodel = Word2Vec(processedText)
wordembeddings = []
wordembeddingsword2Vec = []
for text in processedText:
    #embeddingsFast = np.vstack([np.mean([fastmodel.wv[word] for word in text], axis=0)])
    #embeddingsword = np.vstack([np.mean([word2vecmodel.wv[word] if word in word2vecmodel.wv else np.zeros(100) for word in text], axis=0)])
    embeddingsFast =[fastmodel.wv[word] for word in text]
    averageembedding = sum(embeddingsFast) / len(embeddingsFast) if embeddingsFast else []
    wordembeddings.append(averageembedding)
    #wordembeddings.append(embeddingsFast)    
    #wordembeddingsword2Vec.append(embeddingsword)
    
#wordembeddings = np.array(wordembeddings)
#wordembeddingsword2Vec = np.array(wordembeddingsword2Vec)
data['textFast'] = wordembeddings
#data['textword2vec'] = wordembeddingsword2Vec
sequences = [np.array(embedding) for embedding in wordembeddings]

paddedsequences = padsequences(sequences, dtype='float32')

XtrainF, XtestF, ytrain, ytest = traintestsplit(paddedsequences, data['Label'], testsize=0.35, randomstate=0)

#word2vecmodel.save("word2vec_model.bin")
#fastmodel.save("fasttext_model.bin")
#XtrainWV, XtestWV, ytrain, ytest = traintestsplit(data['textword2vec'], data['Label'], testsize=0.3, randomstate=42)
#XtrainF, XtestF, ytrain, ytest = traintestsplit(wordembeddings, data['Label'], testsize=0.3, randomstate=0)
print(XtrainF.shape)
print(XtestF.shape)
print(ytrain.shape)
print(ytest.shape)

s = SVC(verbose=1).fit(XtrainF,ytrain)
ypred = s.predict(XtestF)
accuracy = accuracyscore(ytest, ypred)
precision = precisionscore(ytest, ypred)
recall = recallscore(ytest, ypred)
f1 = f1score(ytest, ypred)
#rocauc = rocaucscore(ytest, ypredproba)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
#print("ROC AUC:", rocauc)
cm = confusionmatrix(ytest, ypred)
classes = ['No skill', 'Skill']
plotconfusionmatrix(cm, classes,"SVC")
plt.show()
rc = RandomForestClassifier().fit(XtrainF,ytrain)
ypred = rc.predict(XtestF)
precision = precisionscore(ytest, ypred)
recall = recallscore(ytest, ypred)
f1 = f1score(ytest, ypred)
#rocauc = rocaucscore(ytest, ypredproba)



print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
cm = confusionmatrix(ytest, ypred)
classes = ['No Skill', 'Skill']
print("here")
plotconfusionmatrix(cm, classes,"RandomForest")
plt.show()

classifiers = [
    {
        'name': "SVC",
        'classifier': SVC(),
        'paramgrid': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly']
        }
    },
    {
        'name': "RandomForest",
        'classifier': RandomForestClassifier(),
        'paramgrid': {
            'nestimators': [100, 200, 300, 400, 500],
            'maxdepth': [None, 10, 20]
        }
    },
]


for classifier in classifiers:
    
    pipeline = RandomizedSearchCV(
        classifier['classifier'],
        paramdistributions=classifier['paramgrid'],
        cv=5,
        scoring='accuracy',
        niter=10,
        randomstate=42
    )

    pipeline.fit(XtrainF, ytrain)
    accuracy = pipeline.score(XtestF, ytest)
    print("Test Accuracy:", accuracy)
    ypred = pipeline.predict(XtestF)
    ypredproba = pipeline.predict(XtestF)
    accuracy = accuracyscore(ytest, ypred)
    
    # Precision
    precision = precisionscore(ytest, ypred)
    
    # Recall
    recall = recallscore(ytest, ypred)
    
    # F1 Score
    f1 = f1score(ytest, ypred)
    
    # ROC AUC
    rocauc = rocaucscore(ytest, ypredproba)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC:", rocauc)
    cm = confusionmatrix(ytest, ypred)
    classes = ['Class 0', 'Class 1']
    plotconfusionmatrix(cm, classes,"")
    plt.show()


print(pipeline.cvresults)



results = []
"""

for classifier in classifiers:
    pipeline = GridSearchCV(classifier['classifier'], classifier['paramgrid'], cv=5, scoring='accuracy', verbose=10)
    pipeline.fit(XtrainF, ytrain)
    
    bestparams = pipeline.bestparams
    bestscore = pipeline.bestscore
    
    ypred = pipeline.predict(XtestF)
    accuracy = accuracyscore(ytest, ypred)
    precision = precisionscore(ytest, ypred)
    recall = recallscore(ytest, ypred)
    
    results.append({'Classifier': type(classifier['classifier']).name,
                    'Best Parameters': bestparams,
                    'Best Score': bestscore,
                    'Test Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall})

resultsdf = pd.DataFrame(results)


plt.figure(figsize=(10, 6))
sns.lineplot(data=resultsdf, x='Classifier', y='Test Accuracy', marker='o', label='Accuracy')
sns.lineplot(data=resultsdf, x='Classifier', y='Precision', marker='o', label='Precision')
sns.lineplot(data=resultsdf, x='Classifier', y='Recall', marker='o', label='Recall')
plt.xticks(rotation=45)
plt.title("Classifier Performance Comparison")
plt.xlabel("Classifier")
plt.ylabel("Metric Value")
plt.legend()
plt.tightlayout()
plt.show()


classifiers = [
    {
        'name': "SVC",
        'classifier': SVC(),
        'paramgrid': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly']
        }
    },
    {
        'name': "RandomForest",
        'classifier': RandomForestClassifier(),
        'paramgrid': {
            'nestimators': [100, 200, 300, 400, 500],
            'maxdepth': [None, 10, 20]
        }
    },
]

numplots = len(classifiers)
fig, axes = plt.subplots(numplots, figsize=(10, 6 * numplots))

for idx, classifierinfo in enumerate(classifiers):
    classifiername = classifierinfo['name']
    classifier = classifierinfo['classifier']
    paramgrid = classifierinfo['paramgrid']
    
    results = []

    for params in GridSearchCV(classifier, paramgrid, cv=5, verbose=10).getparams()['paramgrid']:
        clf = classifier.setparams(**params)  
        clf.fit(XtrainF, ytrain)

        ypred = clf.predict(XtestF)
        accuracy = accuracyscore(ytest, ypred)
        precision = precisionscore(ytest, ypred)
        recall = recallscore(ytest, ypred)

        results.append({'Parameters': params, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall})


    resultsdf = pd.DataFrame(results)

    sns.scatterplot(data=resultsdf, x='Parameters', y='Accuracy', ax=axes[idx], marker='o', label='Accuracy')
    sns.scatterplot(data=resultsdf, x='Parameters', y='Precision', ax=axes[idx], marker='s', label='Precision')
    sns.scatterplot(data=resultsdf, x='Parameters', y='Recall', ax=axes[idx], marker='^', label='Recall')
    axes[idx].settitle(f'{classifiername} - Parameter-wise Performance')
    axes[idx].setxlabel('Parameter')
    axes[idx].setylabel('Metric Value')
    axes[idx].setxticks(range(len(paramgrid)))
    axes[idx].setxticklabels([str(p) for p in paramgrid], rotation=45)
    axes[idx].legend()

plt.tightlayout()
plt.show()
"""