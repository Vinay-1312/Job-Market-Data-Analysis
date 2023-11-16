
"""
Created on Tue Jul 25 11:41:25 2023

@author: vinay
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouettescore
from gensim.models import Word2Vec,FastText
import numpy as np 
from sklearn.manifold import TSNE
import joblib
import datetime
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
word2vecmodel = Word2Vec.load("word2veccbow_model.bin")
fasttextmodel = FastText.load('fasttextcbow_model.bin')
#word2vecmodel = Word2Vec.load('word2veccbowmodel3.bin')
#fasttextmodel = FastText.load('fasttextcbowmodelResume.bin')
#fasttextmodel = FastText.load('fasttextcbowmodelResume.bin')
#word2vec2model = Word2Vec.load('word2veccbowmodel.bin')
#fasttextmodel = FastText.load('fasttextcbowmodel3CBOW.bin')
vocab = list(word2vecmodel.wv.indextokey)
embeddings = np.array([word2vecmodel.wv[word] for word in vocab])
word2vecvocab = set(word2vecmodel.wv.indextokey)
word2vecembeddings = [word2vecmodel.wv[word] for word in word2vecvocab]
fasttextvocab = set(fasttextmodel.wv.indextokey)
fasttextembeddings = [fasttextmodel.wv[word] for word in fasttextvocab]
allvocab = word2vecvocab.union(fasttextvocab)
allembeddings = np.concatenate([word2vecembeddings, fasttextembeddings],axis=0)
minclusters = 2
maxclusters = 10
silhouettescores = []

"""

neigh = NearestNeighbors(nneighbors=2)
neigh.fit(allembeddings)
distances,  = neigh.kneighbors(allembeddings)

distances = np.sort(distances[:, 1], axis=0)


plt.plot(distances)
plt.xlabel('Points')
plt.ylabel('Distance')
plt.title('Distances to Nearest Neighbor')
plt.show()

kneepoint = np.argmax(np.diff(distances)) + 1
optimaleps = distances[kneepoint]

print("Optimal Epsilon:", optimaleps)
dbscan = DBSCAN(eps=15, minsamples=10)
clusterlabels = dbscan.fitpredict(allembeddings)


joblib.dump(dbscan, 'dbscanmodel.pkl')


uniquelabels = np.unique(clusterlabels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(uniquelabels)))

for label, color in zip(uniquelabels, colors):
    if label == -1: 
        color = 'black'
    
    classmembermask = (clusterlabels == label)
    clustermembers = allembeddings[classmembermask]
    
    plt.scatter(clustermembers[:, 0], clustermembers[:, 1], s=50, c=color, label='Cluster {}'.format(label))

plt.legend()
plt.title('DBSCAN Clustering')
plt.show()

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
kvalues = 10  
neigh = NearestNeighbors(nneighbors=kvalues)
neigh.fit(allembeddings)
distances,  = neigh.kneighbors(allembeddings)
distances = np.sort(distances, axis=0)
distances = distances[:, -1]

plt.plot(distances)
plt.xlabel('Points')
plt.ylabel('Distance')
plt.title('K-Distance Plot')
plt.show()



"""
"""
for numclusters in range(minclusters, maxclusters + 1):
    kmeans = KMeans(nclusters=numclusters, randomstate=42)
    clusterlabels = kmeans.fitpredict(allembeddings)
    silhouetteavg = silhouettescore(allembeddings, clusterlabels)
    silhouettescores.append(silhouetteavg)
inertias = []

for numclusters in range(minclusters, maxclusters + 1):
    kmeans = KMeans(nclusters=numclusters, randomstate=42)
    kmeans.fit(allembeddings)
    inertias.append(kmeans.inertia)

plt.plot(range(minclusters, maxclusters + 1), inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (within-cluster sum of squares)')
plt.title('Elbow Method for Optimal Number of Clusters')

plt.show()
  
plt.plot(range(minclusters, maxclusters + 1), silhouettescores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal Number of Clusters')
plt.show()
"""
#kmeans = KMeans(nclusters=4, randomstate=42)
#joblib.dump(kmeans, 'kmeansmodel4.pkl')
#clusterlabels = kmeans.fitpredict(allembeddings)


class clustering:
    def fileName(self,title):
        imageextension= '.png'
        imagename = title + "Clustering"
        todaydate = datetime.datetime.now().strftime("%Y-%m-%d")  
        newfilename = f"{imagename}{todaydate}{imageextension}"
        return newfilename
    def plotclusterswithtext(self,data, labels, clustercenters,lablels2,skills,title,densityscores,pp):
        
        tsne = TSNE(ncomponents=2, randomstate=42,perplexity=pp)
        datatsne = tsne.fittransform(np.array(data))
    
        #plt.figure(figsize=(40, 40))
        colors = ['red', 'pink', 'black', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        legendlabels = [] 
        uniquelabels = np.unique(lablels2)
        categorymap = {0: 'Skills1', 1: 'Skills2', 2: 'Skill3',3:'Skills4'}

        plt.figure(figsize=(40, 40))
        for label in uniquelabels:
           clusterdata = datatsne[labels == label]
           clusterdensity = np.mean(densityscores[labels == label]) 
           
           clusterdata = datatsne[labels == label]
           clusterskills = np.array(skills)[labels == label]
           plt.scatter(clusterdata[:, 0], clusterdata[:, 1], c=colors[label], edgecolor='k', s=100, label=categorymap[label])
           for i in range(len(datatsne)):
               
                plt.scatter(datatsne[i, 0], datatsne[i, 1], c=colors[lablels2[i]],s=100)
                plt.annotate(str(labels[i]), xy=(datatsne[i, 0], datatsne[i, 1]), xytext=(5, 2),
                             textcoords='offset points', ha='right', va='bottom',fontsize=30)
           #legendlabels.append(f'{categorymap[label]} ({", ".join(clusterskills)})')
        densitylegendlabels = [f'Density Cluster {label}: {np.mean(densityscores[labels == label]):.2f}' for label in uniquelabels]
        #print(densitylegendlabels)
        plt.legend(densitylegendlabels, title="Cluster Densities", fontsize=25, titlefontsize='large', loc='upper right')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('K-Means Clustering with t-SNE and Text Annotations'+title)
        plt.legend(title="Clusters", fontsize=25, titlefontsize='large', loc='upper left')
        
        plt.subplotsadjust(right=0.7)
        plt.grid(True)
         
        plt.savefig("Plots/"+self.fileName(title))

        plt.show()
    def cluster(self,skills,title):
    #tsne = TSNE(ncomponents=2, randomstate=42,perplexity=5)
        print(f"***Process of Clustering started for {title}***")
        kmeans = joblib.load('kmeansmodel.pkl') #old
        #kmeans = joblib.load("kmeansmodel4.pkl") #new
        cmap = plt.cm.getcmap("tab10",3)
        #skills = ['python', 'java', 'jenkins','docker','communication','leadership']
        pp = 5
        if len(skills)<5:
            pp = len(skills) -1 
        if len(skills)<=20 and len(skills)>=5:
            pp =5
        else:
            pp = 30
        newembeddings = []
        newclusterlabels = []
        
        for word in skills:
            if word in word2vecmodel.wv:
                newembeddings.append(word2vecmodel.wv[word])
            elif word in fasttextmodel.wv:
                newembeddings.append(fasttextmodel.wv[word])
        
        if newembeddings:
            newclusterlabels = kmeans.fitpredict(newembeddings)
        
         
        
        else:
            print("New words not present in Word2Vec or FastText vocabulary.")
        uniqueclusters = np.unique(newclusterlabels)
        #embedding2d = tsne.fittransform(np.array(newembeddings))   
        labels = kmeans.labels
        clustercenters = kmeans.clustercenters
        distances = np.linalg.norm(newembeddings - clustercenters[newclusterlabels], axis=1)
        maxdistance = np.max(distances)
        densityscores = 1 - distances / maxdistance
       
        self.plotclusterswithtext(newembeddings,skills,clustercenters,newclusterlabels,skills,title,densityscores,pp)