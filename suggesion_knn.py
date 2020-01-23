import pandas as pd
from sklearn.preprocessing import LabelEncoder

from classifiers.knn import *

#for google drive file upload
df1 = pd.read_csv ("https://raw.githubusercontent.com/Upulee/Sinhala_grammar/master/dataset/Correct_grammar2.csv")

df1 = df1.dropna(how='all', axis='columns')

inputs1 = df1.drop('verb_root',axis='columns')
target1 = df1['verb_root']

subject1 = LabelEncoder()
tense1 = LabelEncoder()
person1 = LabelEncoder()
gender1 = LabelEncoder()
animate1 = LabelEncoder()
number1 = LabelEncoder()
active1 = LabelEncoder()
honorific1 = LabelEncoder()

inputs1['subject_en'] = subject1.fit_transform(inputs1['subject'])
inputs1['gender_en'] = gender1.fit_transform(inputs1['gender'])
inputs1['animate_en'] = animate1.fit_transform(inputs1['animate'])
inputs1['number_en'] = number1.fit_transform(inputs1['number'])
inputs1['person_en'] = person1.fit_transform(inputs1['person'])
inputs1['honorific_en'] = honorific1.fit_transform(inputs1['honorific'])
inputs1['active_en'] = active1.fit_transform(inputs1['active'])
inputs1['tense_en'] = tense1.fit_transform(inputs1['tense'])

encoded_inputs1 = inputs1[['subject_en','tense_en','gender_en','animate_en','number_en','person_en','active_en','honorific_en','pattern']]

knn_x1 = encoded_inputs1
cat_recommendation_data1 = []
for row in knn_x1.values:
    data_row = list(map(float, row))
    cat_recommendation_data1.append(data_row)

k_recommendations1 = 3

def get_suggession(vector, sentence): ##, sentence, verb_suffix
    tenseslist = [2,1,0] ## 
    tl = ['present', 'future', 'past']

    if vector[1] == 'ප්‍රථම':
        s_e = subject1.transform(['3rd_person'])
    else:
        s_e = subject1.transform([vector[6]])

    v = [
        s_e[0],
        gender1.transform([vector[2]])[0],
        animate1.transform([vector[3]])[0],
        number1.transform([vector[0]])[0],
        person1.transform([vector[1]])[0],
        honorific1.transform([vector[4]])[0],
        active1.transform([vector[5]])[0],
        0,
        vector[6]
    ]
    
    #verb_root1.transform([vector[21]])
    #print(tense1.inverse_transform([1]),tense1.inverse_transform([2]))
    #print(len(v))
    suggestions = []

    for t in tenseslist:
        
        
        v[7] = t

        recommendation_indices1, _ = knn(
            cat_recommendation_data1, v, k=k_recommendations1,
            distance_fn=euclidean_distance, choice_fn=lambda x: None
            )
        for rlist in recommendation_indices1:
            if cat_recommendation_data1[rlist[1]][7] == t:
                suggested_root = target1[rlist[1]]

                verb_suffix = get_verbsuffix(sentence[-1])

                suggested_verb = create_verb(verb_suffix[1], suggested_root)
                s = sentence
                s[-1] = suggested_verb

                s_array = [t,u" ".join(s),suggested_root]

                suggestions.append(s_array)

            # if cat_recommendation_data1[rlist[0][1]][7] == t
            #     print( v,'->',cat_recommendation_data1[recommendation_indices1[0][1]],target1[recommendation_indices1[2][1]])
    return suggestions
################################################
################################################

def create_verb(suffix,root):
  if root[0] == 'එ': 
    return (suffix + '‍ෙ' + root[1:])
  elif root[0] == 'ඊ':
    return (suffix + 'ී' + root[1:])
  elif root[:1] == 'ඌ':
    return (suffix + 'ූ' + root[1:])
  elif root[:1] == 'ආ':
    return (suffix + 'ා' + root[1:])
  elif root[0] == 'ඉ':
    return (suffix + 'ි' + root[1:])
  elif root[0] == 'ඒ':
    return (suffix + '‍ෙ' +  '්'+ root[1:])
  else:
    return (suffix + root[1:])

################################################
################################################

### get the verb root
#########################
### get the verb root
#########################
def get_verbsuffix(w): # w: verb
  #return w[-5:-3] 
  ## uththma purusha
  feature = []

  if w[-2:] == 'මි':
    if w[-3] == 'ෙ':
      if w[-5:-3] == '්න':
        feature = ['අන්නෙමි',w[:-6],'න්නෙමි']
      else:
        x = 'ෙ' + w[-2:]
        feature =  ['එමි',w[:-3],x]
    
    else:
      feature =  ['අමි',w[:-2]]
  
  elif w[-2:] == 'මු':
    if w[-3] == 'ෙ':
      if w[-5:-3] == '්න':
        feature =  ['අන්නෙමු',w[:-6]]
      else:
        feature =  ['එමු',w[:-3]]
    
    else:
      feature =  ['අමු',w[:-2] ] 
  elif w[-1] == 'හ':
    if w[-2] == 'ී':
      feature = ['ඊහ',w[:-2]]
    elif w[-2] == 'ූ':
      feature = ['ඌහ',w[:-2]]
    elif w[-2] == 'ා':
      feature = ['ආහ',w[:-2]]
    else:
      return ['අහ',w[:-1]]
  elif w[-2:] == 'ති':
    if w[-3] == 'ෙ':
      feature = ['එති',w[:-3]]
    elif  w[-3] == 'ි':
      feature = ['ඉති',w[:-3]]
    else:
      feature = ['අති',w[:-2]]
  elif w[-1] == 'ය':
    if len(w) > 5:
      #return w[-5:]
      if w[-5:] == "න්නේය":
        feature = ['අන්නේය',w[:-5]]
      elif w[-5:] == 'න්නෝය':
        feature = ['අන්නෝය',w[:-5]]
      elif w[-5:] == 'න්නාය':
        feature = ['අන්නාය',w[:-5]]
      else: return w
    elif w[-2] == 'ා':
      feature = ['ආය',w[:-2]]
    elif w[-2] == 'ී':
      feature = ['ඊය',w[:-2]]
    elif w[-2] == 'ේ':
      feature = ['ඒය',w[:-2]]
    elif w[-2] == 'ෝ':
      feature = ['ඕය',w[:-2]]
  elif w[-2:] == 'යි':
    feature = ['අයි',w[:-2]]
  elif w[-2:] == 'හි':
    if len(w) > 6:
      if w[-6:] == 'න්නෙහි':
        feature = ['අන්නෙහි',w[:-6]]
    elif w[-3] == 'ෙ':
      feature = ['එහි',w[:-3]]
  elif w[-2:] == 'හු':
    if len(w) > 6:
      if w[-6:] == 'න්නෙහු':
        feature = ['අන්නෙහු',w[:-6]]
    elif w[-3] == 'ෙ':
      feature = ['එහු',w[:-3]]
    else:
      feature = ['අහු',w[:-2]]
  return feature