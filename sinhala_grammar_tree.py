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

#for google drive file upload
file = 'https://raw.githubusercontent.com/Upulee/Sinhala_grammar/master/dataset/Sinhala_grammar%20-%20Sinhala_grammar.csv'
df = pd.read_csv(file)

df = df.dropna(how='all', axis='columns')

inputs = df.drop('is_correct',axis='columns')
target = df['is_correct']

"""" Label encoding  """""

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

"""" End of Label encoding  """""

encoded_inputs = inputs[['subject_en','tense_en','gender_en','animate_en','number_en','verb_root_en','person_en','honorific_en','active_en']]

train_X, test_X, train_y, test_y = train_test_split(encoded_inputs, target, test_size=0.3, random_state=1)

""" Data reshaping for KNN classifier """


"""" KNN classifier """
