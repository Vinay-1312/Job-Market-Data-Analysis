from WebScraper import webScrapper
#from NLP import NLP
import pandas as pd
import sqlite3

def data_collection(title):
    try:
        description,location,region,salary,title,links,remote,jobType = webScrapper(title)
        print("***Web Scraping Completed***")
      
        dict1 = {'description': description, 'location': location, 'salary': salary,'title':title,'Job type':jobType,"remote":remote,'Applylink':links}  
        
        
        file = "JobsData.db"
          
    
        conn = sqlite3.connect(file)
        cursor_obj = conn.cursor()
        print("Database Sqlite3.db formed.")
        
        minimum = min([len(description),len(location),len(salary),len(title),len(jobType),len(remote),len(links)])
        for i in range(0,minimum):
            sql = 'Insert Into jobs (title,description,location,region,salary,jobType,remote,links) Values(?,?,?,?,?,?,?,?)'
            cursor_obj.execute(sql,(title[i],description[i],location[i],region[i],salary[i],jobType[i],remote[i],links[i]))
            conn.commit()
           # print("here")
        #processedData = NLP(title,description,location,region,salary,jobType,remote)
    except Exception as e:
       print(e) 
  
#data_collection("Python")
"""
df = pd.DataFrame(dict1) 
df = df.transpose() 
# saving the dataframe 
df.to_csv('All jobs.csv',mode='a')     
"""