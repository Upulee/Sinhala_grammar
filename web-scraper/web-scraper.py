from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import string

driver = webdriver.Chrome(executable_path='F:/BSc. Research/PYTHON/chromedriver.exe')

paragraph=[] #Sinhala paragarph

driver.get("http://www.dinamina.lk/2020/01/18/%E0%B6%B4%E0%B7%94%E0%B7%80%E0%B6%AD%E0%B7%8A/88428/%E0%B6%B1%E0%B7%9C%E0%B6%BB%E0%B7%9C%E0%B6%A0%E0%B7%8A%E0%B6%A0%E0%B7%9D%E0%B6%BD%E0%B7%99-%E0%B6%9C%E0%B7%90%E0%B6%A7%E0%B6%BD%E0%B7%94-%E0%B7%80%E0%B7%92%E0%B7%83%E0%B6%B3%E0%B6%B1%E0%B7%8A%E0%B6%B1-%E0%B6%A2%E0%B6%B1%E0%B6%B4%E0%B6%AD%E0%B7%92-%E0%B6%B8%E0%B7%90%E0%B6%AF%E0%B7%92%E0%B7%84%E0%B6%AD%E0%B7%8A-%E0%B7%80%E0%B7%99%E0%B6%BA%E0%B7%92")

content = driver.page_source
soup = BeautifulSoup(content,features="html.parser")

#for a in soup.findAll('div', attrs={'class':'field field-name-body field-type-text-with-summary field-label-hidden'}):
     #para=a.find('p')
     #paragraph.append(para.text)
for a in soup.findAll('div', attrs={'class':'field field-name-body field-type-text-with-summary field-label-hidden'}):
    for x in a.findAll('p'):
        paragraph.append(x.get_text().split("."))
    
df = pd.DataFrame({'Paragraphs':paragraph}) 
df.to_csv('paragraph.csv', index=False, encoding="utf-8")
#print(df)

