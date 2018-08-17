# # Dataset boolean2: sentences conjoined by and
# 
# Generating sentences of the form
# 
# - 1) **c is P and d is Q, c is not P** (contradiction)
# 
# - 2) **c is P and d is Q, d is not Q** (contradiction)
# 
# - 3) **c is P and d is Q, e is (not) W** (non-contradiction)
# 
# - 4) **c is P and d is Q, c (d) is not Q(P)** (non-contradiction)


import numpy as np
import pandas as pd
from word_lists import name_list, positive_personality_list, condition_list
import os
import itertools

template1 = itertools.product(name_list, positive_personality_list)
template1 = list(template1)
template2 = itertools.product(name_list, condition_list)
template2 = list(template2)
upper_bound = 11000 / 4



np.random.shuffle(template1)
all_sentences_1 = []
for i in range(int(upper_bound)):
    car, cdr = template1[i]
    caar, cdrr = car, cdr
    while caar == car and cdrr == cdr:
        new_i = np.random.choice(len(template1))
        caar, cdrr = template1[new_i]
    all_sentences_1.append("{} is {} and {} is {}, {} is not {}".format(car,cdr,caar, cdrr,car,cdr))

all_sentences_1 = [sentence.split(",") + [1] for sentence in all_sentences_1]

np.random.shuffle(template2)
all_sentences_2 = []
for i in range(int(upper_bound)):
    car, cdr = template2[i]
    caar, cdrr = car, cdr
    while caar == car and cdrr == cdr:
        new_i = np.random.choice(len(template2))
        caar, cdrr = template2[new_i]
    all_sentences_2.append("{} is {} and {} is {}, {} is not {}".format(car,cdr,caar,cdrr,caar,cdrr))

all_sentences_2 = [sentence.split(",") + [1] for sentence in all_sentences_2]

np.random.shuffle(template1)
np.random.shuffle(template2)
all_sentences_3 = []
for i in range(int(upper_bound)):
    car, cdr = template1[i]
    caar, cdrr = car, cdr
    while caar == car and cdrr == cdr:
        new_i = np.random.choice(len(template1))
        caar, cdrr = template1[new_i]
    third_i = np.random.choice(len(template2))
    caaar, cdrrr = template2[third_i]
    if i % 2 == 0:
        my_not = "is"
    else:
        my_not = "is not"
    sentence = "{} is {} and {} is {}, {} {} {}".format(car,cdr,caar,cdrr,caaar,my_not,cdrrr)
    all_sentences_3.append(sentence)

all_sentences_3 = [sentence.split(",") + [0] for sentence in all_sentences_3]



np.random.shuffle(template1)
np.random.shuffle(template2)
all_sentences_4 = []
for i in range(int(upper_bound)):
    car, cdr = template1[i]
    caar, cdrr = car, cdr
    while caar == car and cdrr == cdr:
        new_i = np.random.choice(len(template2))
        caar, cdrr = template2[new_i]
    if i%2 == 0:
        caaar, cdrrr = car, cdrr
    else:
        caaar, cdrrr = caar, cdr
    sentence = "{} is {} and {} is {}, {} is not {}".format(car,cdr,caar,cdrr,caaar,cdrrr)
    all_sentences_4.append(sentence)

all_sentences_4 = [sentence.split(",") + [0] for sentence in all_sentences_4]


np.random.shuffle(all_sentences_1)
np.random.shuffle(all_sentences_2)
np.random.shuffle(all_sentences_3)
np.random.shuffle(all_sentences_4)

size1 = len(all_sentences_1)
size2 = len(all_sentences_2)
size3 = len(all_sentences_3)
size4 = len(all_sentences_4)

all_sentences = all_sentences_1 + all_sentences_2 + all_sentences_3 + all_sentences_4
size = len(all_sentences)


sentence_1 = [triple[0] for triple in all_sentences]
sentence_2 = [triple[1] for triple in all_sentences]
label = [triple[2] for triple in all_sentences]


df_dict = {"sentence1": sentence_1,
           "sentence2": sentence_2,
           "label": label}

df = pd.DataFrame(df_dict)
df = df[["sentence1", "sentence2", "label"]]
df = df.sample(frac=1).reset_index(drop=True)

df_train = df.iloc[:10000]
df_test = df.iloc[10000:]

if not os.path.exists("./data"):
    os.makedirs("data/")

df_train.to_csv("data/boolean2_train.csv", index=False)
df_test.to_csv("data/boolean2_test.csv", index=False)
