from urllib.request import urlopen as ureq
import requests as req
from bs4 import BeautifulSoup as soup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException
from urllib.error import HTTPError
import re
import os
import csv
import pandas as pd
import senti_analysis as senti  # module made for performing the sentiment analysis

#URL of wikipedia page with list of all actors
my_url= 'https://en.wikipedia.org/wiki/List_of_Bollywood_actors'

page_html= ureq(my_url).read()
ureq(my_url).close()

page_soup = soup(page_html,"html.parser") #parsing as html data

actor_letter = page_soup.findAll('div',{'class':'div-col columns column-width'})
temp =[]
names=[]
for a in actor_letter:
    temp.append(a.findAll('a'))
for b in temp:
    for c in b:
        names.append(c.text)
        
#removing duplicate names from list of names
names = list(dict.fromkeys(names)) 

# =============================================================================
# removing last 5 elements from the list which were included due the 
# lack of uniform html structure
# =============================================================================
for x in range (5):
    names.pop(-1)

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get('https://en.wikipedia.org/wiki/List_of_Bollywood_actors')
actor_images = {}
actor_data = {}

#Grabbing the images of the Actors and their information
for name in names:
    try:
        element = driver.find_element_by_link_text(name)
    except NoSuchElementException as exception:
        actor_images[str(name)]='null'
        continue
    element.click()
    url_each = driver.current_url
    try:
        each_page_html = ureq(url_each).read()
        ureq(url_each).close()
        each_page_soup = soup(each_page_html,"html.parser")
    except HTTPError as err:
        actor_images[str(name)]='null'
        print("HTTPError :"+ name)
        driver.back()
        continue
    temp2 = each_page_soup.findAll('table',{'class':'infobox biography vcard'})
    text =""
    abc = each_page_soup.find('div',class_ = 'mw-parser-output')
    try:
        for paragraph in abc.find_all('p'):
            text += paragraph.text
            text = re.sub(r'\[[0-9]*]',' ',text)
            text = re.sub(r'\s+',' ',text)
            text = text.lower()
            text = re.sub(r'\d',' ',text)
            text = re.sub(r'\s+',' ',text)
            actor_data[str(name)] = text
    except AttributeError as error:
        actor_data[str(name)] = 'null'
        print('Attribute error:' + name )
    try:
        img_html = temp2[-1].findAll('img')
    except IndexError:
        print("IndexError :"+ name)
        actor_images[str(name)]='null'
        driver.back()
        continue
    if img_html!= None:
        for image in img_html:
            actor_images[str(name)]=image['src']
    else:
        actor_images[str(name)]='null'
        driver.back()
        continue
    driver.back()
driver.close()

# Making directory to store the actor images
os.mkdir('Celeb_Photos')
for index in actor_images:
    if actor_images[index] != 'null':
        img_data = req.get("https:"+actor_images[index]).content
        with open("Celeb_Photos\\" + str(index)+'.jpg','wb+') as f:
            f.write(img_data)
    else:
        print('no image for' + str(index))
        f.close
        
#making csv file to store the data of actors
with open('actor_data.csv','w',newline='', encoding='utf8') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['Actor Name','Information'])
    for index in actor_data:
        thewriter.writerow([str(index),str(actor_data[index])])


# =============================================================================
# finding sentiment of each actor and storing it in another 
# csv file named actor_Sentiment_data
# =============================================================================
df = pd.read_csv("actor_data.csv")
senti_data = {}
with open('actor_Sentiment_data.csv','w',newline='', encoding='utf8') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['Actor Name','Sentiment Classification','Confidence Score'])
    for x in range(len(df['Information'])):
        df['Actor Name'][x]
        a = senti.sentiment(str(df['Information'][x]))[0]
        b = senti.sentiment(str(df['Information'][x]))[1]
        c = senti_data.update({a:(b,c)})
        thewriter.writerow([df['Actor Name'][x],senti.sentiment(str(df['Information'][x]))[0],senti.sentiment(str(df['Information'][x]))[1]])



        

    
    



