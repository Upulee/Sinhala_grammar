import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import polyglot
from polyglot.text import Text,Word
import re

#for google drive file upload
file = 'https://raw.githubusercontent.com/Upulee/Sinhala_grammar/master/dataset/Sinhala_grammar%20-%20Sinhala_grammar.csv'
df = pd.read_csv(file)

df = df.dropna(how='all', axis='columns')

df.head()

inputs = df.drop('is_correct',axis='columns')
target = df['is_correct']

inputs[:5]

subject = LabelEncoder()
tense = LabelEncoder()
person = LabelEncoder()
gender = LabelEncoder()
animate = LabelEncoder()
number = LabelEncoder()
verb_root = LabelEncoder()
honorific = LabelEncoder()

inputs['subject_en'] = subject.fit_transform(inputs['subject'])
inputs['tense_en'] = tense.fit_transform(inputs['tense'])
inputs['person_en'] = person.fit_transform(inputs['person'])
inputs['gender_en'] = gender.fit_transform(inputs['gender'])
inputs['animate_en'] = animate.fit_transform(inputs['animate'])
inputs['number_en'] = number.fit_transform(inputs['number'])
inputs['verb_root_en'] = verb_root.fit_transform(inputs['verb_root'])
inputs['honorific_en'] = honorific.fit_transform(inputs['honorific'])

encoded_inputs = inputs.drop(['subject','tense','gender','animate','number','verb_root','person','honorific'],axis='columns')

model = tree.DecisionTreeClassifier()

train_X, test_X, train_y, test_y = train_test_split(encoded_inputs, target, test_size=0.3, random_state=1)

train_X[['honorific_en','subject_en']]

kfold = model_selection.KFold(n_splits=10, random_state=100)

model_kfold = tree.DecisionTreeClassifier()
classification_tree = model_kfold.fit(encoded_inputs,target)

results_kfold = model_selection.cross_val_score(classification_tree, encoded_inputs,target, cv=kfold)
print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))


# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
classification_tree = model.fit(train_X,train_y)
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classification_tree, test_X,test_y,
                                 display_labels=['Correct','Incorrect'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

#sni classification_tree = model.fit(train_X,train_y)

#print(model.predict([[33,1,0,1,1,0,18,0]]))

#print(classification_tree.score(test_X,test_y))

list(encoded_inputs.columns.values)

from POStagger import *

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

     
#getPerson("පියා")


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

  #return w[-3] 
x = get_verbroot("දුන්නාය")
x

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
  
  #return postag

getGender("කුඹුර")

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

print(getAnimate("අපි"))

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
  
print(isHonorific("මෑනියන්දෑ"))

### 
def checkGrammar(sentence):
  x = sentence
  w = x.split(" ")

  sentence_pattern = getSentencePattern(sentence)

  if len(sentence_pattern[1]) == 1:
    s = w[0]
    v = w[-1]
    
  v_r = get_verbroot(v)

  s_singular = isSingular(s)
  s_p = getPerson(s)
  s_g = getGender(s)
  s_a = getAnimate(s)
  s_h = isHonorific(s)

  ### encoded values
  if s_p[3] == 'ප්‍රථම':
    s_e = subject.transform(['3rd_pserson'])
  else:
    s_e = subject.transform([s])

  g_e = gender.transform([s_g])
  a_e = animate.transform([s_a])
  n_e = number.transform([s_singular])
  v_e = verb_root.transform([v_r])
  p_e = person.transform([s_p[3]])
  h_e = honorific.transform([s_h])

  tense = [0,1,2] # athith, anagatha , warthamana
  for t in tense:
    print(t,s_p[3],'->',model.predict([[s_e[0],t,p_e[0],g_e[0],a_e[0],n_e[0],v_e[0],h_e[0]]])[0])


### pattern recognition module
def getSentencePattern(tagged_array):
  if len(tagged_array) > 1 and (tagged_array[0] == 'PRP' or tagged_array[0] == 'NNC') and tagged_array[-1] == 'VFM': 
    if tagged_array[1] == 'POST': ##මම සමග කමල් ගෙදර යමු
      return ['2',[0,2],-1]
    elif tagged_array[1] == 'NNC' and tagged_array[2] == 'NNC': ##මම සැමවිටම යාච්ඤා කරමි && මම ඔහුව පුදුම කළෙමි
      return ['4',[0],-1]
    elif tagged_array[1] == 'NNC' and tagged_array[2] == 'PRP': ##ඔහු ඊයේ එය මිලට ගත්තේය
      return ['5',[0],-1]
    elif tagged_array[1] == 'DET' and tagged_array[2] == 'POST': ##මම ඒ පිළිබඳ කතා කරන්නෙමි
      return ['6',[0],-1]
    elif tagged_array[1] == 'PRP' and tagged_array[2] == 'DET' and tagged_array[3] == 'POST': ##ඇය මට ඒ පිළිබඳ විස්තර කළාය
      return ['7',[0],-1]
    elif tagged_array[1] == 'PRP' and tagged_array[2] == 'POST': ##ඇය මා සමඟ තරඟ කරයි
      return ['8',[0,1],-1]
    elif tagged_array[1] == 'RP' and tagged_array[3] == 'RP':  ##මම ද ඔබ ද ගෙදර යමු
      return ['3',[0,2],-1]
    elif tagged_array[1] == 'PRP': ##මම එය නැරඹුවෙමි && මම එය ඊයේ යැව්වෙමි && අපි ඔවුන්ට පහරදෙමු && අපි ඔවුන්ට ආරාධනා කළෙමු
      return ['3',[0],-1] 
    else: return ['1',[0],-1]  #ඔබ ලීවෙමි && මම කතා කළෙමි

    #elif ta
  return 'high five' 

getSentencePattern(posstaggedSentence("ඇය මා සමඟ තරඟ කරයි"))

