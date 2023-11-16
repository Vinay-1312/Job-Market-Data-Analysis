from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from helium import *
import pandas as pd
import datetime
import time
from selenium.webdriver.support import expected_conditions as EC

def webScrapper(jobTitle):
    print("*** Web scraping stared***")
    jobTitle = jobTitle.strip().replace(" ",'-')
    url=""
    links = []
    title = []
    description = []
    salary = []
    location = []
    jobType = []
    remote = []
    region = []
    c=0
    #chrome_options = Options()
    #chrome_options.add_argument("--disable-cookies") 
    #driver = webdriver.Chrome()
    try:
        for i in range(0,5):
          
            if i==0:
                
                url =   'https://www.reed.co.uk/jobs/'+jobTitle+'-jobs-in-united-kingdom'
                #driver.get(url)
                #wait = WebDriverWait(driver, 10)
                #cookies_prompt = wait.until(EC.presence_of_element_located((By.ID, "onetrust-button-group-parent"))) 
                print("here")
                #accept_button = cookies_prompt.find_element(By.ID, "onetrust-accept-btn-handler")
                #accept_button.click() 
               
                print("here")
            else:
                url =   'https://www.reed.co.uk/jobs/'+jobTitle+'-jobs-in-united-kingdom?pageno='+str(i+1) 
                driver.get(url)
           
            driver=start_chrome(url, headless=True)
            #//*[@id="text-input-what"]
                    
            
            driver.implicitly_wait(2)
            #driver.find_element_by_xpath('//form[@id="jobsearch"]')
            #elements = driver.find_elements_by_xpath('/html/body/div[2]/div/div[5]/div[1]/div[3]/div[1]/section//*')#.send_keys("software engineer",Keys.ENTER)
            elements = driver.find_elements_by_xpath('/html/body/div[1]/div[4]/div/div[3]/main//*')
            
            
                
            
            for item in elements:
                #driver.get(url);
                #print(item.text)
                #WebDriverWait(driver, 20)
               
                if item.tag_name =='a' and item.get_attribute('data-element')=='job_title':
                    title.append(item.text)
                    #print(item.get_attribute('title') )
                    WebDriverWait(driver, 10)
                    new_url = item.get_attribute('href')
                    links.append(new_url)
                    #driver.get(new_url);
                    #driver.execute_script("arguments[0].click();",item)
                    #item.click()
                if item.tag_name=='ul' and item.get_attribute("class").lower()=='pagination':
                    continue
                """
                if (item.getTagName().equals("input")):
                
                    print("here")
                """
            """    
            driver.find_element_by_id("text-input-where").send_keys("united kingdom",Keys.ENTER)
            search_button = driver.find_element_by_xpath('//*[@id="jobsearch"]/button')
            search_button.click()
            """
        print("Fetched Links =",links)
        description = []
        salary = []
        location = []
        jobType = []
        remote = []
        region = []
        c=0
        for link in links:
            
            remoteJob = 0
            permenant =0
            fulltime = 0
            parttime = 0
            contract =0 
            temp = 0
            driver.get(link)
            WebDriverWait(driver, 10)
            url = driver.current_url
            
            #elements = driver.find_elements_by_xpath('/html/body/div[2]/div/div[5]/div[1]/div[2]/article/div/div[2]/div/div[3]//*')
            elements = driver.find_elements_by_xpath('/html/body//*')
            
            #divElements = driver.find_elements_by_xpath('/html/body/div[1]/div/div[5]/div[1]/div[2]/article/div/div[2]//*')
            #for ele in divElements:
            #    print(ele.text)
            #ele = driver.find_element_by_xpath('/html/body/div[1]/div/div[5]/div[1]/div[2]/article/div/div[2]/div/div[2]/div/div[1]/span')
            for item in elements:
                #print(item.tag_name)
                #print(item.text)
                
                #print(item.text)
              
                if item.tag_name=='div':
                    if 'remote' in item.get_attribute('class'):
                        text =   driver.execute_script("return arguments[0].childNodes[2].textContent", item);
                        if 'work from home' in text.lower().strip() or 'remote' in text.lower().strip():
                            remoteJob==1
                if item.tag_name == 'span':
                    if item.get_attribute('data-qa') =='salaryLbl':
                        salary.append(item.text)
                    
                    if item.get_attribute('itemprop')=='addressLocality' :
                        location.append(item.text)
                    if item.get_attribute('data-qa')=='localityLbl':
                         region.append(item.text)
                    
                    if item.get_attribute('itemprop') =='employmentType' or item.get_attribute('data-qa')=='jobTypeLbl':
                        
                        b1Text = driver.execute_script("return arguments[0].childNodes[1].textContent", item);
                        b2Text = driver.execute_script("return arguments[0].childNodes[3].textContent", item);
                        jobType.append(b1Text.strip()+','+b2Text.strip())
                           
                       
                    
                    if item.get_attribute('itemprop')=='description':
                        description.append(item.text)
                 
            if remoteJob==1:
                remote.append('remote')
            else:
                remote.append('onsite')
                
            if len(salary)<len(description):
                salary.append('competative')
            if len(location)<len(description):
                location.append('other')
            if len(region)<len(description):
                region.append('other')
            if len(remote)<len(description):
                remote.append('not given')
            if len(jobType)<len(description):
                jobType.append('Permanent,full-time')
                    
                
            
            """
            if permenant ==1 and fulltime==1:
                jobType.append("permanent,full-time")
            if permenant ==1 and parttime==1:
                jobType.append("permanent,part-time")
            if contract ==1 and fulltime==1:
                jobType.append("contract,full-time")
            if contract ==1 and parttime==1:
                   jobType.append("contract,part-time")
            """
             
    
            time.sleep(2)
        driver.quit()
        print("Process of Web scrapping completed")  
        return description,location,region,salary,title,links,remote,jobType
    except:
      
        return description,location,region,salary,title,links,remote,jobType

    
