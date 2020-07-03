import sent2vec
from numpy import dot
from numpy.linalg import norm
import numpy as np
import os

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def divide_sentence(direction, file):
    
    f = open('./' + direction + file, 'r', encoding='utf-8-sig')
    
    li = []

    for line in f:
        li.append(line.strip())

    sentence = []
    count = 0
    
    for i in range(len(li)-1):
        etc = []
        etc.append(li[i])
        etc.append(li[i+1])
        sentence.append(etc)
        count += 1

    return sentence, count

def topic_change_cosine(li, mean_cosine, counts):
    change_count = 0
    count = 0
    person_topic_change = []
    
    for i in li:
        count += 1
        if i < mean_cosine:
            change_count += 1
            
        if count == counts[0]:
            person_topic_change.append(change_count/counts[0])
            change_count = 0
            count = 0
            del counts[0]
    return person_topic_change

file = open('파일', 'r', encoding='utf-8-sig') # 같은 토픽을 가지고 있는 문장 쌍의 파일

sentence = []
label = []

for idx, line in enumerate(file):
    line = line.strip().split('\t')
    etc = []
    etc.append(line[0].strip())
    etc.append(line[1].strip())
    sentence.append(etc)
    
model = sent2vec.Sent2vecModel()
model.load_model('Sent2Vec')

cosine = []

for i in range(len(sentence)):
    left = model.embed_sentence(line[0])
    right = model.embed_sentence(line[1])
    cosine.append(cos_sim(left[0],right[0]))

mean_cosine = sum(cosine)/len(cosine)  # 코사인 유사도 기준 값

path = "1학년 폴더"
file_list = os.listdir(path)
file_list_grade1 = [file for file in file_list if file.endswith(".txt")]

path = "3학년 폴더"
file_list = os.listdir(path)
file_list_grade3 = [file for file in file_list if file.endswith(".txt")]

path = "5학년 폴더"
file_list = os.listdir(path)
file_list_grade5 = [file for file in file_list if file.endswith(".txt")]

grade1_sentence = []
grade1_count = []

for age_file in file_list_grade1:
    direction = '1학년 폴더/'
    sent, count = divide_sentence(direction, age_file)
    grade1_sentence += sent
    grade1_count.append(count)

grade3_sentence = []
grade3_count = []

for age_file in file_list_grade3:
    direction = '3학년 폴더/'
    sent, count = divide_sentence(direction, age_file)
    grade3_sentence += sent
    grade3_count.append(count)

grade5_sentence = []
grade5_count = []

for age_file in file_list_grade5:
    direction = '5학년 폴더/'
    sent, count = divide_sentence(direction, age_file)
    grade5_sentence += sent
    grade5_count.append(count)

grade1_cosine = []

for line in grade1_sentence:
    left = model.embed_sentence(line[0])
    right = model.embed_sentence(line[1])
    grade1_cosine.append(cos_sim(left[0],right[0]))

grade3_cosine = []

for line in grade3_sentence:
    left = model.embed_sentence(line[0])
    right = model.embed_sentence(line[1])
    grade3_cosine.append(cos_sim(left[0],right[0]))

grade5_cosine = []

for line in grade5_sentence:
    left = model.embed_sentence(line[0])
    right = model.embed_sentence(line[1])
    grade5_cosine.append(cos_sim(left[0],right[0]))

grade_1 = topic_change_cosine(grade1_cosine, mean_cosine, grade1_count)
grade_3 = topic_change_cosine(grade3_cosine, mean_cosine, grade3_count)
grade_5 = topic_change_cosine(grade5_cosine, mean_cosine, grade5_count)

print('1학년의 토픽 변화율 : ', sum(grade_1)/len(grade_1))
print('3학년의 토픽 변화율 : ', sum(grade_3)/len(grade_3))
print('5학년의 토픽 변화율 : ', sum(grade_5)/len(grade_5))
