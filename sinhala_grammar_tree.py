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

#for google drive file upload
file = 'https://raw.githubusercontent.com/Upulee/Sinhala_grammar/master/dataset/Sinhala_grammar%20-%20Sinhala_grammar.csv'
df = pd.read_csv(file)

df = df.dropna(how='all', axis='columns')

inputs = df.drop('is_correct',axis='columns')
target = df['is_correct']

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

    #print(disp.confusion_matrix)

##plt.show()

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

print(getSentencePattern(posstaggedSentence("ඇය මා සමඟ තරඟ කරයි")))
