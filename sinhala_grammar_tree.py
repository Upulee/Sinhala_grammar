import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.model_selection import KFold

from sklearn import tree

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

from sinhala.nlp.feature_extraction import *
from sinhala.nlp.sentence_pattern import *

from classifiers.knn import *

from suggesion_knn import get_suggession

#for google drive file upload
file = 'https://raw.githubusercontent.com/Upulee/Sinhala_grammar/master/dataset/Grammar_rules.csv'
df = pd.read_csv(file)

df = df.dropna(how='all', axis='columns')

inputs = df.drop('is_correct',axis='columns')
target = df['is_correct']

#"""" Label encoding  """""

subject = LabelEncoder()
tense = LabelEncoder()
person = LabelEncoder()
gender = LabelEncoder()
animate = LabelEncoder()
number = LabelEncoder()
active = LabelEncoder()
honorific = LabelEncoder()
verb_root = LabelEncoder()

inputs['subject_en'] = subject.fit_transform(inputs['subject'])
inputs['tense_en'] = tense.fit_transform(inputs['tense'])
inputs['person_en'] = person.fit_transform(inputs['person'])
inputs['gender_en'] = gender.fit_transform(inputs['gender'])
inputs['animate_en'] = animate.fit_transform(inputs['animate'])
inputs['number_en'] = number.fit_transform(inputs['number'])
inputs['honorific_en'] = honorific.fit_transform(inputs['honorific'])
inputs['verb_root_en'] = verb_root.fit_transform(inputs['verb_root'])
inputs['active_en'] = active.fit_transform(inputs['active'])

#"""" End of Label encoding  """""

encoded_inputs = inputs[['subject_en','tense_en','gender_en','animate_en','number_en','verb_root_en','person_en','honorific_en','active_en']]

train_X, test_X, train_y, test_y = train_test_split(encoded_inputs, target, test_size=0.3, random_state=1)

#""" Data reshaping for KNN classifier """
knn_x = encoded_inputs
cat_recommendation_data = []

## convert column values to float
for row in knn_x.values:
  data_row = list(map(float, row))
  cat_recommendation_data.append(data_row)

#"""" Grammar checking using KNN classifier """
def checkGrammarknn(sentence):
  x = sentence
  w = x.split(" ")

  sentence_pattern = getSentencePattern(posstaggedSentence(sentence))

  if len(sentence_pattern[1]) == 1:
    s = w[0]
    v = w[-1]
  else: ## need to change the implementation
    s = w[0]
    v = w[-1]
    
  v_r = get_verbroot(v)

  s_singular = isSingular(s)
  s_p = getPerson(s)
  s_g = getGender(s)
  s_a = getAnimate(s)
  s_h = isHonorific(s)
  s_active = isActive(s)

  ### encoded values
  if s_p[3] == 'ප්‍රථම':
    s_e = subject.transform(['3rd_person'])
  else:
    s_e = subject.transform([s])

  g_e = gender.transform([s_g])
  a_e = animate.transform([s_a])
  n_e = number.transform([s_singular])
  v_e = verb_root.transform([v_r])
  p_e = person.transform([s_p[3]])
  h_e = honorific.transform([s_h])
  active_e = active.transform([s_active])

  tense = [0,1,2] # athith, anagatha , warthamana

  k_recommendations = 3

  for t in tense:
    recommendation_indices, _ = knn(
      cat_recommendation_data, [s_e[0],t,g_e[0],a_e[0],n_e[0],v_e[0],p_e[0],h_e[0],active_e[0]], k=k_recommendations,
      distance_fn=euclidean_distance, choice_fn=lambda x: None
      )
    #print([s,t,s_g,s_a,s_singular,v_r,s_p[3],s_h],[s_e[0],t,g_e[0],a_e[0],n_e[0],v_e[0],p_e[0],h_e[0],active_e[0]],recommendation_indices,target[recommendation_indices[0][1]])
    #print(t, '->',target[recommendation_indices[0][1]])
    if target[recommendation_indices[0][1]] == 1:
      return [True]
    else: 
      return [False,[s_singular,s_p[3],s_g,s_a,s_h,s_active,s],w]

grammar_result = checkGrammarknn("මම ගෙදර ගියෙමු")
if grammar_result[0] == False:
  #print(grammar_result[2][-1])
  print(get_suggession(grammar_result[1],grammar_result[2]))
else:
  print("correct")