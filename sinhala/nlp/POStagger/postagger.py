import pandas as pd
import numpy as np
import re
import polyglot
from polyglot.text import Text,Word

poss_tag_list = pd.read_csv("https://raw.githubusercontent.com/Upulee/Sinhala_grammar/master/sinhala/nlp/POStagger/data/simple_sample.txt")

#morphological analyzer

#method declaration
########################
########################
def POStagged(searchstring):
   # print(searchstring)
    #Declaration of array
    z = []
    array2D = []

    #File opening in 2D array
    if True:
      i = 0
      for line in poss_tag_list.values:
        line = line[0]
        if 0 < i < len(poss_tag_list.values):
          array2D.append(line.rstrip("\n").split(' '))
          y = ([z[1] for z in array2D if z[0] == searchstring])
          w = ([z[0] for z in array2D if z[0] == searchstring])
        i = i +1
      if not y:
        #print("empty")
        y = "empty"
      else:
        if y[0] == 'PRP':
            Noun = w[0]
            #print('Pronoun ->',Noun)
        elif y[0][0:3] == 'NNC':
            Common_Noun = w[0]
            #y[0] = y[0][0:3]
            #print('Common Noun ->',Common_Noun)
        if y[0] == 'VFM':
            Verb = w[0]
            #print('Verb ->',Verb)
            words = [Verb]
            for w in words:
              w = Word(w,language="si")
              #print(w)
              #print("Morphem of word: {:<20}{}".format(w,w.morphemes))
        elif y[0] == 'RB':
            Adverb = w[0]
            #print('Adverb ->',Adverb)
        elif y[0] == 'RP':
            Nipatha = w[0]
            #print('Nipatha ->',Nipatha)
        
        return y[0]
    return "Not Found"

#separating white spaces from words and returning words only in an array

print(POStagged("සීයා"))

import string

## Post tag the given sentence
############################
############################
############################

def posstaggedSentence(sentence):
  translator = str.maketrans('', '', string.punctuation)
  st = sentence.translate(translator)
  new_str = st.split()

  q = []
  q = new_str
  taggedSentence = []
  for line in q: 
    taggedSentence.append(POStagged(line))
    #w = Word(line, language='si')
    #taggedSentence.append(w.morphemes)
  return taggedSentence