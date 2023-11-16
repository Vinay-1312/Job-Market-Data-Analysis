

import plotly.express as px
import re
import nltk
from nltk.tokenize import senttokenize, wordtokenize
import sqlite3
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
import seaborn as sns
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from spacy.matcher import PhraseMatcher
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from skillNer.generalparams import SKILLDB
from skillNer.skillextractorclass import SkillExtractor
from sklearn.featureextraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText
from gensim import similarities
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import time
import json
import folium
import googlemaps
import random
from folium.plugins import MarkerCluster
from collections import Counter
from clustering3 import clustering
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import typing
import datetime
    
from sklearn.featureextraction.text import TfidfVectorizer
endpoint = 'https://apiskills12.cognitiveservices.azure.com/'
key =''

nlp = spacy.load("encoreweblg")
filepath = 'skilllist.txt'
f = open('skilldbrelax20.json')
Skill = json.load(f)

ListOfSkills = []

with open(filepath, 'r') as file:
    ListOfSkills = list(set(file.read().splitlines()))

skillextractor = SkillExtractor(nlp, SKILLDB, PhraseMatcher)
stopwords = stopwords.words('english')
remote2 = {'remote':0,'hybrid':0,"onsite":0}
skills = {}
data = pd.readcsv("data.csv")
def mainfunction(title):
    print("Process Of data Processing Started.")
    def fileName(Type):
        imageextension= '.png'
        imagename = title + "" + Type
        todaydate = datetime.datetime.now().strftime("%Y-%m-%d")  
        newfilename = f"{imagename}{todaydate}{imageextension}"
        return newfilename
    def locationVsSalaryScatter(locations,salary,Type=None):
        try: 
          imageextension= '.png'
          imagename = title + "Salary Location Scatter"
          todaydate = datetime.datetime.now().strftime("%Y-%m-%d")  
          newfilename = f"{imagename}{todaydate}{imageextension}"
          tempdic = {}
          finalDic = {}
          data = {
              'Location':locations,
              'Salary': salary
          }
          
          for x,y in zip(locations,salary):
              if x not in tempdic.keys():
                  tempdic[x] = []
                  tempdic[x].append(y)
              else:
                  tempdic[x].append(y)
        
          for x,y in tempdic.items():
              finalDic[x] = {'min':min(y),'max':max(y),'avg':np.average(y)}
              
          locations = list(finalDic.keys())
          sorteditems = sorted(finalDic.items(), key=lambda x: len(x[1]), reverse=True)
    
  
          locations = [item[0] for item in sorteditems]
          ##print("here")
          minimumsalaries = [data['min'] for data in finalDic.values()]
          maximumsalaries = [data['max'] for data in finalDic.values()]
          averagesalaries = [data['avg'] for data in finalDic.values()]
    
        
         
          plt.figure(figsize=(20, 20))
        
         
          xpositions = np.arange(len(locations))
          for x in xpositions:
              plt.plot([x, x], [minimumsalaries[x], maximumsalaries[x]], color='gray', linestyle='-', linewidth=2)
        
          markersize = 100
        
          plt.scatter(xpositions, maximumsalaries, color="red", marker='o', s=markersize, label="Maximum Salary")
          plt.scatter(xpositions, minimumsalaries, color="green", marker='o', s=markersize, label="Minimum Salary")
          plt.scatter(xpositions, averagesalaries, color="blue", marker='o', s=markersize, label="Average Salary")
        
      
          plt.xticks(xpositions, locations,rotation=90)
          plt.xlabel("Location")
          plt.ylabel("Salary")
          plt.title("Salary Analysis by Location")
          plt.legend()
          plt.savefig("Plots/"+newfilename)
          plt.show()
        except Exception as e:
            print(e)
       
    def processtext(text):
        
        sentences = senttokenize(text)
        
       
        #nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        
  
        #nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        
        processedsentences = []
        for sentence in sentences:
           
            words = wordtokenize(sentence)
            
            # Stop word removal and lemmatization
            filteredwords = [word.lower() for word in words if word.lower() not in stop_words]
            
           
            processedsentence = ' '.join(filteredwords)
            processedsentences.append(processedsentence)
        
        return processedsentences
    def plotJobType(jobType):
        data =   {
         'jobType':jobType
         }
        imageextension= '.png'
        imagename = title + "Job type"
        todaydate = datetime.datetime.now().strftime("%Y-%m-%d")  
        newfilename = f"{imagename}{todaydate}{imageextension}"
        df = pd.DataFrame(data)
        jobtypeCount = df['jobType'].valuecounts()
        plt.figure(figsize=(16, 16))
        jobtypeCount.plot(kind='bar')
        plt.xticks(fontsize=20  ) 
        plt.xlabel('Type of job')
        plt.ylabel('Count')
        plt.title(title+" Job Type")
        plt.savefig("Plots/"+newfilename)
        plt.show()
    
    
    def extractexperience(description):
        experiencekeywords = ['years of experience', 'experience required', 'minimum experience']
        extractedexperience = None
    
        for keyword in experiencekeywords:
            pattern = r'(\d+)\s?{}'.format(keyword)
            match = re.search(pattern, description, flags=re.IGNORECASE)
            if match:
                extractedexperience = match.group(1)
                break
    
        return extractedexperience
    
    def extracttechnology(description, technologykeywords):
        extractedtechnology = []
    
        for keyword in technologykeywords:
            escapedkeyword = re.escape(keyword)
            if re.search(r'\b{}\b'.format(escapedkeyword), description, flags=re.IGNORECASE):
                extractedtechnology.append(keyword)
    
        return extractedtechnology
    
    
    def processjobdescriptions(jobdescriptions, technologykeywords):
        extracteddata = []
    
        for description in jobdescriptions:
            #print(description)
            experience = extractexperience(description)
            technology = extracttechnology(description, technologykeywords)
            extracteddata.append((experience, technology))
    
        return extracteddata
     
    
    def plotSalaryLocation(locations,salary,Type=None):
        imageextension= '.png'
        imagename = title + "Salary Location"
        todaydate = datetime.datetime.now().strftime("%Y-%m-%d")  
        newfilename = f"{imagename}{todaydate}{imageextension}"
        tempdic = {}
        finalDic = {}
        data = {
            'Location':locations,
            'Salary': salary
        }
        for x,y in zip(locations,salary):
            if x not in tempdic.keys():
                tempdic[x] = []
                tempdic[x].append(y)
            else:
                tempdic[x].append(y)
                
        for x,y in tempdic.items():
            finalDic[x] = {'min':min(y),'max':max(y),'avg':np.average(y)}
        ##print(finalDic)
        locations = list(finalDic.keys())
        minsalaries = [data['min'] for data in finalDic.values()]
        maxsalaries = [data['max'] for data in finalDic.values()]
        avgsalaries = [data['avg'] for data in finalDic.values()]
        
        fig, ax = plt.subplots(figsize=(20, 20))
    
   
        barwidth = 0.2
        
     
        barpositions = np.arange(len(locations))
        

        plt.bar(barpositions, minsalaries, width=barwidth, label='Minimum Salary')
        plt.bar(barpositions + barwidth, maxsalaries, width=barwidth, label='Maximum Salary')
        plt.bar(barpositions + 2 * barwidth, avgsalaries, width=barwidth, label='Average Salary')
        
     
        plt.xticks(barpositions + barwidth, locations, rotation=90,fontsize=15)
        
      
        plt.ylabel('Salaries')
        
       
        plt.title(title + ' Location-wise Salaries')
        
       
        plt.legend()
        
        
        plt.savefig("Plots/"+newfilename)
        plt.tightlayout()
        plt.show()
        
    
    def replacewithmaxoccurrences(lst):
        
        counts = {}
        for num in lst:
            if num != -1:
                counts[num] = counts.get(num, 0) + 1
    
       
        maxoccurrencesnum = max(counts, key=counts.get)
    
       
        replacedlst = [num if num != -1 else maxoccurrencesnum for num in lst]
    
        return replacedlst
    
    def processSalaries(salary):
        salariesProcessed = []
        nlp = spacy.load('encorewebsm')
        
        for sal in salary:
            match = re.search(r'\d+', sal)
            ##print(sal)
            ##print(salariesProcessed)
            if match:        
            
                pattern = ""
                
                
                pattern = r'£([\d,.]+)'
                matches = re.findall(pattern, sal)
                if matches:
                    if len(matches)==2:
                        if 'per day' in sal.lower():
                            lowerBound = int(float(matches[0].replace(',','')))
                            upperBound = int(float(matches[1].replace(',','')))
                            salariesProcessed.append(((lowerBound+upperBound)/2)*365)
                            
                        elif 'per month' in sal.lower():
                            lowerBound = int(float(matches[0].replace(',','')))
                            upperBound = int(float(matches[1].replace(',','')))
                            salariesProcessed.append(((lowerBound+upperBound)/2)*12)
                        elif 'per annum' in sal.lower():
                            lowerBound = int(float(matches[0].replace(',','')))
                            upperBound = int(float(matches[1].replace(',','')))
                            salariesProcessed.append(((lowerBound+upperBound)/2))
                    else:
                           
                           value = int(float(matches[0].replace(',','')))
                           if 'per day' in sal.lower():
                               
                               salariesProcessed.append(value*365)
                           elif 'per month' in sal.lower():
                                
                                salariesProcessed.append(value*12)
                           else:
                               salariesProcessed.append(value)
              
            else:
                salariesProcessed.append(-1)
       
        return salariesProcessed
    def salaryVsExperience(entryLevel,midLevel,seniorLevel):
        levels = ['entry level', 'mid level', 'senior level']
        numvalues = 50
        np.random.seed(42)
        minsalaries = [min(entryLevel), min(midLevel), min(seniorLevel)]
        avgsalaries = [np.mean(entryLevel), np.mean(midLevel), np.mean(seniorLevel)]
        maxsalaries = [max(entryLevel), max(midLevel), max(seniorLevel)]
        
        plt.figure(figsize=(16, 16))
        
        barwidth = 0.2
        index = np.arange(len(levels))
        
        plt.bar(index - barwidth, minsalaries, barwidth, label='Minimum Salary')
        plt.bar(index, avgsalaries, barwidth, label='Average Salary')
        plt.bar(index + barwidth, maxsalaries, barwidth, label='Maximum Salary')
        
   
        plt.xlabel('Levels')
        plt.ylabel('Salaries')
        plt.title(title + 'Salary Comparison for Different Levels')
        plt.xticks(index, levels)
        plt.grid(True)
        
    
        plt.legend()
        
      
        plt.tightlayout()
        plt.savefig("Plots/"+fileName('SalaryExperience'))
        plt.show()
    
    def locationSalary(locations,salary):
        tempdic = {}
        finalDic = {}
        data = {
            'Location':locations,
          }
        data2 = pd.DataFrame(data)
        for x,y in zip(locations,salary):
            if x not in tempdic.keys():
                tempdic[x] = []
                tempdic[x].append(y)
            else:
                tempdic[x].append(y)
                
        for x,y in tempdic.items():
            finalDic[x] = {'min':min(y),'max':max(y),'avg':np.average(y)}
        ##print(finalDic)
        locations = list(finalDic.keys())
        
        gmaps = googlemaps.Client(key='AIzaSyBn4nTK7Wkcd2fHx4kUS6uzhM1rMUqQbh0')
        def cord(city):
            geocoderesult = gmaps.geocode(city + ", UK")
            if geocoderesult:
                latitude = geocoderesult[0]['geometry']['location']['lat']
                longitude = geocoderesult[0]['geometry']['location']['lng']
                return latitude,longitude
            return None
    
        #print(data['Location'])
        data2['Coordinates'] = data2['Location'].apply(cord)
        data2= data2.dropna(subset=['Coordinates'])
        #print(data2.columns)
      
        mapcenter = data2['Coordinates'].iloc[0]
        m = folium.Map(location=mapcenter, zoomstart=6)
        
        for index, row in data2.iterrows():
            city,(lat, lon)  = row
            if city in finalDic.keys():
                avgsalary = finalDic[city]['avg']
                minsalary = finalDic[city]['min']
                maxsalary = finalDic[city]['max']
                popuptext = f"{city}<br>Avg Salary: £{avgsalary}<br>Min Salary: £{minsalary}<br>Max Salary: £{maxsalary}"
                folium.Marker(location=(lat, lon), popup=popuptext).addto(m)
    
        m.save("Plots/"+title+'salarymap.html')
    
    def locationsCountMap(locations,locationcounts):
        occurrences = locationcounts
        #print(occurrences)
      
        gmaps = googlemaps.Client(key='')
        
        
        mapcenter = [54.970000, -2.460000]  
        mymap = folium.Map(location=mapcenter, zoomstart=6)
        
        maxoccurrences = max(occurrences)
        
        
        def getcolor(count):
            normalizedcount = count / maxoccurrences
            color = f"#{int(normalizedcount * 255):02x}0000"  
            return color
        
        
        for location in locations:
            count = occurrences[location]
            color = getcolor(count)
            geocoderesult = gmaps.geocode(location)
            if geocoderesult:
                latitude = geocoderesult[0]['geometry']['location']['lat']
                longitude = geocoderesult[0]['geometry']['location']['lng']
                folium.Marker(
                    location=[latitude, longitude],
                    popup=f"{location}: {count} occurrences",
                    icon=folium.Icon(color=color),
                ).addto(mymap)
               
    
            else:
                print(f"Failed to geocode {location}")
        mymap.save("Plots/"+title+"occurrencemap.html")
    
    def NLP(title):    
        print("***The process of NLP started.***")
        global remote2,skills,nlp,ListOfSkills
        #vectorizer = TfidfVectorizer() 
        allDescription =[]
        try:
            
            tokenizeddescription = []
            lemmatizer = WordNetLemmatizer()
            stemmer = PorterStemmer()
            searchstr = ('%'+title+'%',)
            file = "JobsData.db"
            conn = sqlite3.connect(file)
            cursorobj = conn.cursor()
            title = []
            description = []
            location = []
            region = []
            salary = []
            jobType = []
            remote = []
            cursorobj.execute("SELECT * FROM jobs where title like ? ", searchstr)
            processedNLP = []  
            nlp1 = []
            rows = cursorobj.fetchall()
            finalData = []
            for row in rows:
                  title.append(row[0])
                  #allDescription.append(row[1])
                  tokenizeddescription.append(senttokenize(row[1]))
                  location.append(row[2])
                  region.append(row[3])
                  salary.append(row[4])
                  jobType.append(row[5])
                  remote.append(row[6])
                  processedsentences = processtext(row[1])
                  processedNLP.append(processedsentences)
            for sentence in processedNLP:
               desc = ""
               for sen in sentence:
                   desc+=sen
                
                ##print(skills)
               cleandescription = re.sub(r'\bacting[\w\s]+agency\b', '', desc, flags=re.IGNORECASE)
               allDescription.append(cleandescription)
               finalData.append(skillextractor.annotate(cleandescription))   
            return finalData,location ,region,salary,jobType,allDescription
         
        except Exception  as e:
          #print("Database Sqlite3.db not formed.")
          #print(e)
          return finalData,location,region,salary,jobType,allDescription
      
    def getsimilarity(skill1, skill2):
        #if skill1 in word2vecmodel.wv.keytoindex and skill2 in word2vecmodel.wv.keytoindex:
         #   return word2vecmodel.wv.similarity(skill1, skill2)
        #elif skill1 in fasttextmodel.wv.keytoindex and skill2 in fasttextmodel.wv.keytoindex:
        if  skill1 in word2vec2model.wv.keytoindex and skill2 in word2vec2model.wv.keytoindex: 
            return word2vec2model.wv.similarity(skill1, skill2)  
        else:
               return fasttextmodel.wv.similarity(skill1, skill2)
    
    def plotSkillsTops(skillscounts,title1):
        sortedskills = sorted(skillscounts.items(), key=lambda x: x[1], reverse=True)
        topskills = dict(sortedskills[:10])
        skills = list(topskills.keys())
        counts = list(topskills.values())
        plt.figure(figsize=(10, 6))
        plt.bar(skills, counts, color='skyblue')
        plt.xlabel('Skills')
        plt.ylabel('Count')
        plt.title(title+' Top 10 Skills using '+title1)
        plt.xticks(rotation=45, ha='right')
        plt.tightlayout()
        plt.savefig("Plots/"+fileName('Top 10 skills using '+title1))
        plt.show()
          
    def plotSkills(embedding,skillsList,title1):
        x=[]
        y=[]
        for value in embedding:
            x.append(value[0])
            y.append(value[1])
        plt.figure(figsize=(25, 25))
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(skillsList[i],
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom',fontsize=22)
        plt.title(title+'skills using'+title1)
        plt.savefig("Plots/"+fileName('Skills using'+title1))
        plt.show()
    
    def commonSkills(skills1,skills2):
        commonskillscounts = {}
        for skill in set(skills1.keys()).intersection(skills2.keys()):
            commonskillscounts[skill] = max(skills1[skill], skills2[skill])
        newS1kills = {key: value for key, value in commonskillscounts.items() if value>=3}
        wordcloud = WordCloud(width = 1000, height = 500).generatefromfrequencies(newS1kills)
        
        plt.figure(figsize=(15,8))
        plt.imshow(wordcloud)
        plt.savefig("Plots/"+fileName('Skills Cloud Common'))
        plt.show()
        skillsList = [k for k in list(newS1kills.keys()) if newS1kills[k] >=3] 
        newSkills = {key: value for key, value in newS1kills.items() if value>=3}
        similaritymatrix1 = []
        for i in range(len(skillsList)):
            row = []
            for j in range(len(skillsList)):
                if i == j:
                    row.append(1.0)
                else:
                    similarity = getsimilarity(skillsList[i], skillsList[j])
                    if similarity is not None:
                        row.append(similarity)
                    else:
                        row.append(0.0) 
            similaritymatrix1.append(row)
                        
        tsne1 = TSNE(ncomponents=2, randomstate=42)
        embedding1 = tsne.fittransform(np.array(similaritymatrix1))
        plotSkills(embedding1,skillsList,"Common skills")
        plotSkillsTops(newSkills,"Common skils")
        c1 = clustering()
        c1.cluster(skillsList, title+"Common")
        
    
    finalData,locations,regions,salary,jobType,allDescription = NLP(title)
    
    
    data = {
          'locations':locations,
          'region':regions}
    
    df = pd.DataFrame(data)
    
    locationcounts = df['locations'].valuecounts()
    top10locations = locationcounts.head(10)
    plt.figure(figsize=(16, 16))
    plt.bar(top10locations.index, top10locations.values)
    plt.yticks(range(0, max(top10locations.values) + 1, 5))
    plt.xlabel('Location')
    plt.ylabel('Count')
    plt.title('Location Occurrences')
    for index, value in enumerate(top10locations.values):
        plt.text(index, value + 0.5, str(value), ha='center', va='bottom')
    plt.savefig("Plots/"+fileName('Location Count'))
    plt.show()
    regioncounts = df['region'].valuecounts()
    top10regions = regioncounts.head(10)
    # Plotting the region counts
    plt.figure(figsize=(16, 16))
    top10regions.plot(kind='bar')
    

    plt.xlabel('Region')
    plt.ylabel('Count')
    plt.title('Region Occurrences')
    plt.savefig("Plots/"+fileName('Region Count'))

    plt.show()
    
    df = pd.DataFrame(data)
    keywords = []
    import os
    import openai
    
    experience = {"entry level":0,"Mid Level":0,"Senior Level":0}
    experienceList = []
    jobTyperemote = {"remote":0,"hybrid":0,"onsite":0}
    openai.apikey = ""
    c =0 
    
    for desc in allDescription[0:50]:
        c+=1
        completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo-0301",
          messages=[
            {"role": "user", "content": desc+'''Give me answers for the following questions based on the above job description
             1) Does the above job require experience or not. Give answer as Senior level or entry level or mid level experience
             2) Job is remote or hybrid or onsite.
            if years of experienced is not specified in the first answer then give output as 0. For the second question give only one word answer. Do not metnion the questions in the answers 
            
             '''}
          ]
        )
    
        chatgptOutput = completion.choices[0].message.content.split("\n")
        if 'mid' in chatgptOutput[0].lower():
            experience['Mid Level']+=1
            experienceList.append("Mid Level")
        elif 'senior' in chatgptOutput[0].lower():
              experience['Senior Level']+=1
              experienceList.append("Senior Level")
        elif 'entry' in chatgptOutput[0].lower():
              experience['entry level']+=1
              experienceList.append("Entry Level")
        else:
            experience['entry level']+=1
            experienceList.append("Entry Level")
        if 'remote' in chatgptOutput[1].lower():
            jobTyperemote['remote']+=1
        elif 'hybrid' in chatgptOutput[1].lower():
              jobTyperemote['hybrid']+=1
        elif 'onsite' in chatgptOutput[1].lower():
              jobTyperemote['onsite']+=1
        else:
            jobTyperemote['onsite']+=1
        if c%3 ==0:
            c =0
            time.sleep(80)
    
    labels = list(experience.keys())
    values = list(experience.values())
    plt.figure(figsize=(15, 15))
    plt.bar(labels, values)
    plt.xticks(fontsize=13) 
    plt.xlabel('Experience')
    plt.ylabel('Count')
    plt.title('Job Experience Distribution')
    plt.savefig("Plots/"+fileName('Job experience distribution'))
    plt.show()
    
    
    labels = list(jobTyperemote.keys())
    values = list(jobTyperemote.values())
    plt.figure(figsize=(15, 15))
    plt.bar(labels, values)
    plt.xticks(fontsize=13) 
    plt.xlabel('Job Type')
    plt.ylabel('Count')
    plt.title('Job Type Distribution')
    plt.savefig("Plots/"+fileName('Job type distribution'))
    plt.show()
    
    skill2 = {}
    test = []
    for i in finalData:
            for j in i['results']['fullmatches']:
                          
                                word = j['docnodevalue']              
                                skill2[word] = j['score']
                                if j['score']==1  :
                                    if j['docnodevalue'] not in skills.keys():
                                        
                                        skills[j['docnodevalue']]=1
                                    else:
                                        skills[j['docnodevalue']]+=1
                            
            for t in i['results']['ngramscored']:
                word = t['docnodevalue'] 
                skill2[word] = t['score']
                if t['score']==1:
                        
                            if t['docnodevalue'] not in skills.keys():
                                
                                skills[t['docnodevalue']]=1
                            else:
                                skills[t['docnodevalue']]+=1
            
              
    
         
    
    #print("NER")
    #print(skills)
    newSkills = {key: value for key, value in skills.items() if value>=3}            
    wordcloud = WordCloud(width = 1000, height = 500).generatefromfrequencies(newSkills)
    """
    extracteddata = processjobdescriptions(allDescription, skills.keys())
    
  
    for data in extracteddata:
        #print("Experience:", data["experience"])
        #print("Technology:", data["technology"])
        #print("---")
        
    """
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.savefig("Plots/"+fileName('Skills Cloud NER'))

    plt.show()
    salariesProcessed = replacewithmaxoccurrences(processSalaries(salary))
    plotSalaryLocation(locations,salariesProcessed)
    locationVsSalaryScatter(locations,salariesProcessed)
    locationVsSalaryScatter(regions,salariesProcessed)
    plotJobType(jobType)
    locationsCountMap(locations,locationcounts)
    locationSalary(locations,salariesProcessed)
    entryLevelSalaries = []
    midLevelSalaries = []
    seniorLevelSalaries = []
    """
    for sal,level in zip(salariesProcessed[0:len(experienceList)],experienceList):
        if level == 'Mid Level':
            midLevelSalaries.append(sal)
        elif level=='Senior Level':
            seniorLevelSalaries.append(sal)
        elif level =='Entry Level':
            entryLevelSalaries.append(sal)
    
    """
    #salaryVsExperience(entryLevelSalaries,midLevelSalaries,seniorLevelSalaries)
    #word2vec2model = Word2Vec.load('word2veccbowmodel3.bin')l
    # = FastText.load('fasttextcbowmodel.bin')
    #fasttextmodel = FastText.load('fasttextcbowmodelResume.bin')
    fasttextmodel = FastText.load('fasttextcbow_model.bin')
    word2vec2model = Word2Vec.load('word2veccbow_model.bin')
    #fasttextmodel = FastText.load('fasttextcbowmodel3CBOW.bin')
    
    skillsList = [k for k in list(skills.keys()) if skills[k] >=3] 
    newSkills = {key: value for key, value in skills.items() if value>=3}
    similaritymatrix = []
    
    """
    categories = ['Programming Languages', 'Technology', 'Soft Skills', 'Other Skills']
    skills = pd.readcsv("Skillsdata.csv")
    programming = []
    Technology = []
    soft  = []
    other = []
    isFound = 0
    for skill in skillsList:
        isFound = 0
        for value in df['Programming Languages']:
            if skill in value:
                programming.append(skill)
                isFound = 1
                break
        if isFound==0:
            for value in df['Technology Skills']:
                if skill in value:
                    Technology.append(skill)
                    isFound = 1
                    break
        if isFound==0:
            for value in df['Soft Skills']:
                if skill in value:
                    soft.append(skill)
                    isFound = 1
                    break
        if isFound==0:
            other.append(skill)
    """
    for i in range(len(skillsList)):
        row = []
        for j in range(len(skillsList)):
            if i == j:
                row.append(1.0) 
            else:
                similarity = getsimilarity(skillsList[i], skillsList[j])
                if similarity is not None:
                    row.append(similarity)
                else:
                    row.append(0.0)  
        similaritymatrix.append(row)
  
    
    tsne = TSNE(ncomponents=2, randomstate=42,perplexity=5)
    embedding = tsne.fittransform(np.array(similaritymatrix))
    legendelements = []
    
    
    """
    for i, category in enumerate(categories):
        legendelements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=category))
    # Plot the t-SNE embedding
    """
    plt.figure(figsize=(20, 20))
    markersize = 50
    textoffset = 1 
    #plt.scatter(embedding[:, 0], embedding[:, 1], c='white',alpha=0.5, edgecolors='none')
    """
    
    for i, skill in enumerate(newSkills):
        x, y = embedding[i, 0], embedding[i, 1]
        """
    """
        if i < len(programming):
            categorycolor = cmap(0)  # Category 1 color
            categorylabel = categories[0]
        elif i < len(programming) + len(Technology):
            categorycolor = cmap(1)  # Category 2 color
            categorylabel = categories[1]
        elif i < len(programming) + len(Technology) + len(soft):
            categorycolor = cmap(2)  # Category 3 color
            categorylabel = categories[2]
        else:
            categorycolor = cmap(3)  # Category 4 color
            categorylabel = categories[3]
        """
    """
        #plt.text(x, y + textoffset, skill, fontsize=10, ha='right')
        plt.annotate(skill, (x, y), fontsize=10)
    #plt.legend(labels=categories, loc='upper left', bboxtoanchor=(1, 1))
    #print("here2")
    plt.savefig("tsneplot.png", dpi=100)
    plt.title('t-SNE Plot of Skills')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tightlayout()
    plt.show()  
    """
    x = []
    y = []
    
    skills3 = {}
    textanalyticsclient = TextAnalyticsClient(endpoint, AzureKeyCredential(key))
    
    first = 0
    last = 5
    while last<len(allDescription):
        result = textanalyticsclient.recognizeentities(allDescription[first:last])
        result = [review for review in result if not review.iserror]
        organizationtoreviews: typing.Dict[str, typing.List[str]] = {}
        Skills = []
        for idx, review in enumerate(result):
            for entity in review.entities:
                
                if entity.category == 'Skill' or entity.category=='Product':
                    if entity.confidencescore >=0.75:
                        if entity.text not in skills3.keys():
                            skills3[entity.text] = 1
                        else:
                            skills3[entity.text] += 1
        first+=5
        last+=5
        time.sleep(10)
    #print("Azure")
    #print(skills3)
    skills1List = [k for k in list(skills3.keys()) if skills3[k]>=3]  
    newS1kills = {key: value for key, value in skills3.items() if skills3[key]>=3}
    wordcloud = WordCloud(width = 1000, height = 500).generatefromfrequencies(newS1kills)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.savefig("Plots/"+fileName('Azure Cloud NER'))
    plotSkillsTops(newS1kills,"Azure Analytics")
    plotSkillsTops(newSkills,"Named Entity Recognition")
    """
    extracteddata = processjobdescriptions(allDescription, skills.keys())
    
    
 
    for data in extracteddata:
        #print("Experience:", data["experience"])
        #print("Technology:", data["technology"])
        #print("---")
        
    """
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.title("Azure Analytics")
    
    plt.savefig("Plots/"+fileName('Skills Cloud Azure Analytics'))

    plt.show()
    plt.show()
    similaritymatrix1 = []
    for i in range(len(skills1List)):
        row = []
        for j in range(len(skills1List)):
            if i == j:
                row.append(1.0)
            else:
                similarity = getsimilarity(skills1List[i], skills1List[j])
                if similarity is not None:
                    row.append(similarity)
                else:
                    row.append(0.0) 
        similaritymatrix1.append(row)
                    
    tsne1 = TSNE(ncomponents=2, randomstate=42,perplexity=5)
    embedding1 = tsne1.fittransform(np.array(similaritymatrix1))
    legendelements = []
    
    plotSkills(embedding1,skills1List,"Azure Analytics")
    """
    for value in embedding1:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(20, 20))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(skills1List[i],
        xy=(x[i], y[i]),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')
    plt.title("Using azure analytics")
    plt.show()
    """
    x=[]
    y=[]
    plotSkills(embedding,skillsList,"Named Entity Recognition")
    """
    for value in embedding:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(20, 20))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(skillsList[i],
        xy=(x[i], y[i]),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')
    plt.show()
    """
    c1 = clustering()
    c1.cluster(skillsList,title+"NER")
    c1.cluster(skills1List,title+"Azure")
#mainfunction("data analyst")
