import polyglot
from polyglot.text import Text,Word
import re

from sinhala.nlp.POStagger.postagger import *

## check singularr or plural
############################
############################
def isSingular(w):
  plural_suffixes = ["න්",'ෝ','කරුවන්','වරුන්']
  w = Word(w,language="si")

  for suffix in plural_suffixes:
    if suffix == w.morphemes[-1]:
      return "බහු"
  
  return "ඒක"


import string

## identifying the person of the subject
#####################################
#####################################

def getPerson(r):
  q = ['මම','අපි']
  p = ['තෝ','තී','තොපි','තෙපි','ඔබ','නුඹ','ඔබලා','නුඹලා']

  person_1 = "උත්තම"
  person_2 = "මධ්‍යම"
  person_3 = "ප්‍රථම"
  
  for line in q:
      if(line == r):
        return 'Subject ->',line,'Person ->',person_1   
  for li in p:
    if(li == r):
      return 'Subject ->',li,'Person ->',person_2
   
  return 'Subject ->',r,'Person ->',person_3


### get the verb root
#########################
def get_verbroot(w):
  #return w[-5:-3] 
  ## uththma purusha
  if w[-2:] == 'මි':
    if w[-3] == 'ෙ':
      if w[-5:-3] == '්න':
        return 'අන්නෙමි'
      else:
        return 'එමි'
    
    else:
      return 'අමි'
  
  elif w[-2:] == 'මු':
    if w[-3] == 'ෙ':
      if w[-5:-3] == '්න':
        return 'අන්නෙමු'
      else:
        return 'එමු'
    
    else:
      return 'අමු'  
  elif w[-1] == 'හ':
    if w[-2] == 'ී':
      return 'ඊහ'
    elif w[-2] == 'ූ':
      return 'ඌහ'
    elif w[-2] == 'ා':
      return 'ආහ'
    else:
      return 'අහ'
  elif w[-2:] == 'හි': 
    return 'එහි'
  elif w[-2:] == 'ති':
    if w[-3] == 'ෙ':
      return 'එති'
    elif  w[-3] == 'ි':
      return 'ඉති'
    return 'අති'
  elif w[-1] == 'ය':
    if len(w) > 5:
      if w[-5:] == 'න්නේය':
        return 'අන්නේය'
      elif w[-5:] == 'න්නෝය':
        return 'අන්නෝය'
      elif w[-5:] == 'න්නාය':
        return 'අන්නාය'
    if w[-2] == 'ා':
      return 'ආය'
    elif w[-2] == 'ී':
      return 'ඊය'
    elif w[-2] == 'ේ':
      return 'ඒය'
    elif w[-2] == 'ෝ':
      return 'ඕය'
  elif w[-2:] == 'යි':
    return 'අයි'
  elif w[-2:] == 'හි':
    if len(w) > 6:
      if w[-6:] == 'න්නෙහි':
        return 'අන්නෙහි'
    elif w[-3] == '‍ෙ':
      return 'එහි'
  elif w[-2:] == 'හු':
    if len(w) > 6:
      if w[-6:] == 'න්නෙහු':
        return 'අන්නෙහු'
    elif w[-3] == '‍ෙ':
      return 'එහු'
    else:
      return 'අහු'

### get gender
def getGender(w):
  postag = re.sub(r'[^a-zA-Z]', "", POStagged(w))
  #return postag
  if postag == "PRP": 
    return 'නොමැත'
  if len(postag) > 3:
    g = postag[3]
  else:
    return 'නොමැත'

  if g == "M": 
    return "පුරුෂ"
  elif g == "F": 
    return "ස්ත්‍රී"
  elif g == "N":
    return "නපුංසක"
  


### get animate of the given word
#################################
def getAnimate(w):
  postag = POStagged(w)
  #return postag
  if postag == "PRP": 
    return "ප්‍රාණවාචි"
  if len(postag) > 5:
    a = postag[5]
  else:
    return "අප්‍රාණවාචි"

  if a == 'N':
    return "අප්‍රාණවාචි"
  elif a == 'A':
    return "ප්‍රාණවාචි"


#### check whether the subject is horific or not
##################################################

def isHonorific(w):
  if w[-2:] == 'දෑ':
    return 'y'
  elif w[-2:] == 'ඕ':
    return 'y'
  elif w[-2:] == 'හු':
    return 'y'
  elif w[-2:] == 'ඩි':
    return 'y'
  elif w[-3:] == 'අණු' or w[-3:] == 'අණි' or w[-3:] == 'අණු':
    return 'y'
  elif w[-4:] == 'ණ්ඩි':
    return 'y'
  elif len(w) > 6:
    if w[-6:] == 'වහන්සේ':
      return 'y'
  else:
    return 'n'
  return 'n'

######################## get active or passive
def isActive(w):
  if w == 'මාතා':
    return 'n'
  else:
    return 'y'