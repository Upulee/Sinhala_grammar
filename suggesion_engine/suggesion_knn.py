import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

#for google drive file upload
df = pd.read_csv ("https://raw.githubusercontent.com/Upulee/Sinhala_grammar/master/dataset/Correct_grammar.csv")

df = df.dropna(how='all', axis='columns')

inputs = df.drop('verb_root',axis='columns')
target = df['verb_root']

subject = LabelEncoder()
tense = LabelEncoder()
person = LabelEncoder()
gender = LabelEncoder()
animate = LabelEncoder()
number = LabelEncoder()
active = LabelEncoder()
honorific = LabelEncoder()

inputs['subject_en'] = subject.fit_transform(inputs['subject'])
inputs['tense_en'] = tense.fit_transform(inputs['tense'])
inputs['person_en'] = person.fit_transform(inputs['person'])
inputs['gender_en'] = gender.fit_transform(inputs['gender'])
inputs['animate_en'] = animate.fit_transform(inputs['animate'])
inputs['number_en'] = number.fit_transform(inputs['number'])
inputs['active_en'] = active.fit_transform(inputs['active'])
inputs['honorific_en'] = honorific.fit_transform(inputs['honorific'])

encoded_inputs = inputs.drop(['subject','tense','gender','animate','number','person','active','honorific'],axis='columns')

model = tree.DecisionTreeClassifier()

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(encoded_inputs, target, test_size=0.3, random_state=1)

classification_tree = model.fit(train_X,train_y)



print(model.predict([[1,1,0,1,1,1,1,0]]))
