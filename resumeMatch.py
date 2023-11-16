import os
import PyPDF2
import gensim
from gensim.models import FastText
from nltk.tokenize import wordtokenize
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
import sqlite3
from nltk.tokenize import senttokenize, wordtokenize
import sqlite3
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
import datetime
fasttextmodel = FastText.load('fasttextcbow_model.bin')

def preprocesstext(text):
    tokens = wordtokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    tokens = [token for token in tokens if token not in string.punctuation]
    return tokens
def calculatesimilarity(doc1, doc2):
    tokens1 = preprocesstext(doc1)
    tokens2 = preprocesstext(doc2)
    
    similarity = fasttextmodel.wv.nsimilarity(tokens1, tokens2)
    return similarity


def extracttextfrompdf(pdfpath):
    text = ""
    with open(pdfpath, "rb") as pdffile:
        pdfreader = PyPDF2.PdfReader(pdffile)
        for page in pdfreader.pages:
            text += page.extracttext()
    return text

def main(title,pdfpath):
    print("***Process of Resume Matching Started.***")
    def fileName(Type):
         
         imageextension= '.png'
         imagename = title + "" + Type
         todaydate = datetime.datetime.now().strftime("%Y-%m-%d")  
         newfilename = f"{imagename}{todaydate}{imageextension}"
         return newfilename
    def processtext(text):
        # Sentence tokenization
        sentences = senttokenize(text)
        
     
        #nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        
        # Initialize NLTK's WordNetLemmatizer
        #nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        
        processedsentences = []
        for sentence in sentences:
            # Word tokenization
            words = wordtokenize(sentence)
            
            # Stop word removal and lemmatization
            filteredwords = [word.lower() for word in words if word.lower() not in stop_words]
            processedsentence = ' '.join(filteredwords)
            processedsentences.append(processedsentence)
        return processedsentences
    searchstr = ('%'+title+'%',)
    file = "JobsData.db"
    conn = sqlite3.connect(file)
    cursorobj = conn.cursor()
    
    description = []
    location = []
    region = []
    salary = []
    jobType = []
    remote = []
    allDescription = []
    
    cursorobj.execute("SELECT * FROM jobs where title like ? ", searchstr)
    processedNLP = []  
    nlp1 = []
    rows = cursorobj.fetchall()
    finalData = []
    jobdescriptions = []
    
    for row in rows:
          #title.append(row[0])
          allDescription.append(row[1])
          #processedsentences = processtext(row[1])
          #jobdescriptions.append(processedsentences)
    
    
    pdfpath = "C:\\Users\\vinay\\Downloads\\Vinay DeshmukhCV.pdf" 
    resumetext = extracttextfrompdf(pdfpath)
    resumetokens = preprocesstext(resumetext)
    
    similarityscores = []
    c= 0
    for jobdescription in allDescription:
        c+=1
        jobdescriptiontokens = preprocesstext(jobdescription)
        similarity = calculatesimilarity(" ".join(jobdescriptiontokens), " ".join(resumetokens))
        similarityscores.append(similarity)
        print(f"Similarity with Job Description {c}: {similarity:.2f}")
    
    averagesimilarity = sum(similarityscores) / len(similarityscores)
    print(f"Average Similarity Score: {averagesimilarity:.2f}")
    
    # Display top 10 similarity scores using a bar graph
    topscores = sorted(similarityscores, reverse=True)[:10]
    jobindices = sorted(range(len(similarityscores)), key=lambda k: similarityscores[k], reverse=True)[:10]
    joblabels = [f"Job {i+1}" for i in jobindices]
    plt.Figure(figsize=(13,13))
    plt.bar(joblabels, topscores)
    plt.xticks(rotation=90, ha='right')
    plt.xlabel(f'{title}Job Descriptions')
    plt.ylabel("Similarity Score")
    plt.title("Top 10 Similarity Scores")
    plt.annotate(f"Average Score on all descriptions: {averagesimilarity:.2f}", xy=(0.5, 0.95), xycoords="axes fraction", ha="center")
    plt.savefig("Plots/"+fileName("Resume Matching Score"))
    plt.show()
