# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 20:59:47 2022

@author: Nithya Sheena
"""
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
filmography_url = "https://en.wikipedia.org/wiki/Category:Indian_filmographies"

#This is the function to get BeautifulSoup object for any given URL
def Requested_page(url):                  
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception('Failed to load page {}'.format(url))
    # Parse the `response' text using BeautifulSoup
    doc = BeautifulSoup(response.text, 'html.parser')
    return doc

page=Requested_page(filmography_url)
# This is to get the name of actors/actress.
a_tags = page.find_all('a',{'class': None})
print(a_tags)
len(a_tags)

#This is to get name of actors
titles_a_tag = []
for tag in a_tags:
    if 'filmography' in tag.text:
        #replacing filmography text with empty space
        titles_a_tag.append(tag['title'].replace('filmography',''))
        
titles_a_tag

# Now we need to fetch the URL's of all the filmographies
# I have taken all the filmographies url which has "filmography" in all of them.
link_a_tag = []
for tag in a_tags:
    if 'filmography' in tag.text:
        link_a_tag.append(tag['href'])

link_a_tag

# To get the full link we need to give the base url like below
# we can loop to go through all the actors
filmography_urls = []
for i in range(len(link_a_tag)):
    filmography_urls.append("https://en.wikipedia.org/"+ link_a_tag[i])


#creating a dataframe that contains names of actors & their corresponding urls
Filmography_dict = {
    'Actors Name' : titles_a_tag,
    'Url': filmography_urls,
    
}
#checking the total number of urls in the page
filmography_df=pd.DataFrame(Filmography_dict)
len(filmography_df)
nameOfActor = input('Enter the name of actor\n')  
url='https://en.wikipedia.org//wiki/'+ nameOfActor +'_filmography'
# get_docs fuction will reterive all the tables which has class "Wikitable"
def get_film_page(films_url):
    
    response = requests.get(films_url)
    response.status_code
    if response.status_code == 200 : # check the status_code 
            topic_doc = BeautifulSoup(response.text,'html.parser')
            actor_name = topic_doc.find('h1').text.replace(' filmography','')
            table_tag = topic_doc.find_all('table',{'class': 'wikitable'})
            
    else :
        print('Failed to load page {}'.format(films_url)) # if status_code is not equal to 200
    return actor_name,table_tag

 

name, table = get_film_page(url)
print(table)

# we will covert the given Table in Pandas Dataframe
dfs=[]

for each in table:
    if len(dfs)!=2:
        df=pd.read_html(str(each))
        df=pd.DataFrame(df[0])
        df=df[df.columns[0:3]]
        dfs.append(df)  

dfs.pop(0)    
df=df.sort_values(by=['Year'], ascending=False)
print(df)

    
  
    
   

   