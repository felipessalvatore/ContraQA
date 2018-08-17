
# coding: utf-8

# # Dataset boolean3: NP conjoined by and
# 
# Generating sentences of the form
# 
# - 1) **c has traveled to X and Y, c didn't travel to X** (contradiction)
# - 1) **c went to X and Y, c didn't go to X** (contradiction)
# - 1) **c has visited X and Y, c didn't visit X** (contradiction)
# 
# 
# - 2) **c has traveled to X and Y, c didn't travel to Y** (contradiction)
# - 2) **c went to X and Y, c didn't go to Y** (contradiction)
# - 2) **c has visited X and Y, c didn't visit Y** (contradiction)
# 
# 
# - 3) **c has traveled to X and Y, c didn't travel to W ** (non-contradiction)
# - 3) **c went to X and Y, c didn't go to W** (non-contradiction)
# - 3) **c has visited X and Y, c didn't visit W** (non-contradiction)
# 
# 
# - 4) **c has traveled to X and Y, d didn't travel to X (Y)** (non-contradiction)
# - 4) **c went to X and Y, d didn't go to X (Y)** (non-contradiction)
# - 4) **c has visited X and Y, d didn't visit X (Y)** (non-contradiction)
# 
# 
# - 5) **c and d have traveled to X, c didn't travel to X** (contradiction)
# - 5) **c and d went to X, c didn't go to X** (contradiction)
# - 5) **c and d have visited X, c didn't visit X** (contradiction)
# 
# 
# - 6) **c and d have traveled to X, d didn't travel to X** (contradiction)
# - 6) **c and d went to X, d didn't go to X** (contradiction)
# - 6) **c and d have visited X, d didn't visit X** (contradiction)
# 
# 
# - 7) **c and d have traveled to X, c didn't travel to Y ** (non-contradiction)
# - 7) **c and d went to X, c didn't go to Y** (non-contradiction)
# - 7) **c and d have visited X, c didn't visit Y** (non-contradiction)
# 
# 
# - 8) **c and d have traveled to X, e didn't travel to X** (non-contradiction)
# - 8) **c and d went to X, e didn't go to X** (non-contradiction)
# - 8) **c and d have visited X, e didn't visit X** (non-contradiction)


import numpy as np
import pandas as pd
try:
    from word_lists import name_list, city_list
except ImportError:
    from contra_qa.text_generation.word_lists import name_list, city_list

import os
import itertools

def boolean3():

    template1 = itertools.product(name_list, city_list)
    template1 = list(template1)
    upper_bound = 11000/8


    # ### Generating all types of sentences

    # - 1) **c has traveled to X and Y, c didn't travel to X** (contradiction)
    # - 1) **c went to X and Y, c didn't go to X** (contradiction)
    # - 1) **c has visited X and Y, c didn't visit X** (contradiction)


    np.random.shuffle(template1)
    all_sentences_1 = []
    for i in range(int(upper_bound)):
        person, place1 = template1[i]
        place2 = place1
        while place2 == place1:
            new_i = np.random.choice(len(template1))
            _, place2 = template1[new_i]
        if i%3 == 0:
            sentence = "{} has traveled to {} and {},{} didn't travel to {}".format(person, place1, place2, person, place1)
        elif i%3 ==1:
            sentence = "{} went to {} and {},{} didn't go to {}".format(person, place1, place2, person, place1)
        else:
            sentence = "{} has visited {} and {},{} didn't visit {}".format(person, place1, place2, person, place1)
        all_sentences_1.append(sentence)

        
    all_sentences_1 = [sentence.split(",") + [1] for sentence in all_sentences_1]


    # - 2) **c has traveled to X and Y, c didn't travel to Y** (contradiction)
    # - 2) **c went to X and Y, c didn't go to Y** (contradiction)
    # - 2) **c has visited X and Y, c didn't visit Y** (contradiction)


    np.random.shuffle(template1)
    all_sentences_2 = []
    for i in range(int(upper_bound)):
        person, place1 = template1[i]
        place2 = place1
        while place2 == place1:
            new_i = np.random.choice(len(template1))
            _, place2 = template1[new_i]
        if i%3 == 0:
            sentence = "{} has traveled to {} and {},{} didn't travel to {}".format(person, place1, place2, person, place2)
        elif i%3 ==1:
            sentence = "{} went to {} and {},{} didn't go to {}".format(person, place1, place2, person, place2)
        else:
            sentence = "{} has visited {} and {},{} didn't visit {}".format(person, place1, place2, person, place2)
        all_sentences_2.append(sentence)

        
    all_sentences_2 = [sentence.split(",") + [1] for sentence in all_sentences_2]

    # - 3) **c has traveled to X and Y, c didn't travel to W ** (non-contradiction)
    # - 3) **c went to X and Y, c didn't go to W** (non-contradiction)
    # - 3) **c has visited X and Y, c didn't visit W** (non-contradiction)

    # In[5]:


    np.random.shuffle(template1)
    all_sentences_3 = []
    for i in range(int(upper_bound)):
        person, place1 = template1[i]
        place2 = place1
        while place2 == place1:
            new_i = np.random.choice(len(template1))
            _, place2 = template1[new_i]
        place3 = place1
        while place3 == place1 or place3 == place2:
            new_i = np.random.choice(len(template1))
            _, place3 = template1[new_i]
        if i%3 == 0:
            sentence = "{} has traveled to {} and {},{} didn't travel to {}".format(person, place1, place2, person, place3)
        elif i%3 ==1:
            sentence = "{} went to {} and {},{} didn't go to {}".format(person, place1, place2, person, place3)
        else:
            sentence = "{} has visited {} and {},{} didn't visit {}".format(person, place1, place2, person, place3)
        all_sentences_3.append(sentence)

        
    all_sentences_3 = [sentence.split(",") + [0] for sentence in all_sentences_3]


    # - 4) **c has traveled to X and Y, d didn't travel to X (Y)** (non-contradiction)
    # - 4) **c went to X and Y, d didn't go to X (Y)** (non-contradiction)
    # - 4) **c has visited X and Y, d didn't visit X (Y)** (non-contradiction)


    np.random.shuffle(template1)
    all_sentences_4 = []
    for i in range(int(upper_bound)):
        person1, place1 = template1[i]
        place2 = place1
        person2 = person1
        while place2 == place1 and person2 == person1:
            new_i = np.random.choice(len(template1))
            person2, place2 = template1[new_i]
        if i%2 ==0:
            place3 = place1
        else:
            place3 = place2
        if i%3 == 0:
            sentence = "{} has traveled to {} and {},{} didn't travel to {}".format(person1, place1, place2, person2, place3)
        elif i%3 == 1:
            sentence = "{} went to {} and {},{} didn't go to {}".format(person1, place1, place2, person2, place3)
        else:
            sentence = "{} has visited {} and {},{} didn't visit {}".format(person1, place1, place2, person2, place3)
        all_sentences_4.append(sentence)

        
    all_sentences_4 = [sentence.split(",") + [0] for sentence in all_sentences_4]

    # - 5) **c and d have traveled to X, c didn't travel to X** (contradiction)
    # - 5) **c and d went to X, c didn't go to X** (contradiction)
    # - 5) **c and d have visited X, c didn't visit X** (contradiction)

    template1 = itertools.product(name_list, city_list)
    template1 = list(template1)

    np.random.shuffle(template1)
    all_sentences_5 = []
    for i in range(int(upper_bound)):
        person1, place = template1[i]
        person2 = person1
        while person2 == person1:
            new_i = np.random.choice(len(template1))
            person2, _ = template1[new_i]
        if i%3 == 0:
            sentence = "{} and {} have traveled to {},{} didn't travel to {}".format(person1, person2, place, person1, place)
        elif i%3 ==1:
            sentence = "{} and {} went to {},{} didn't go to {}".format(person1, person2, place, person1, place)
        else:
            sentence = "{} and {} have visited {},{} didn't visit {}".format(person1, person2, place, person1, place)
        all_sentences_5.append(sentence)

    all_sentences_5 = [sentence.split(",") + [1] for sentence in all_sentences_5]


    # - 6) **c and d have traveled to X, d didn't travel to X** (contradiction)
    # - 6) **c and d went to X, d didn't go to X** (contradiction)
    # - 6) **c and d have visited X, d didn't visit X** (contradiction)

    np.random.shuffle(template1)
    all_sentences_6 = []
    for i in range(int(upper_bound)):
        person1, place = template1[i]
        person2 = person1
        while person2 == person1:
            new_i = np.random.choice(len(template1))
            person2, _ = template1[new_i]
        if i%3 == 0:
            sentence = "{} and {} have traveled to {},{} didn't travel to {}".format(person1, person2, place, person2, place)
        elif i%3 ==1:
            sentence = "{} and {} went to {},{} didn't go to {}".format(person1, person2, place, person2, place)
        else:
            sentence = "{} and {} have visited {},{} didn't visit {}".format(person1, person2, place, person2, place)
        all_sentences_6.append(sentence)

    all_sentences_6 = [sentence.split(",") + [1] for sentence in all_sentences_6]

    # - 7) **c and d have traveled to X, c (d) didn't travel to Y ** (non-contradiction)
    # - 7) **c and d went to X, c (d) didn't go to Y** (non-contradiction)
    # - 7) **c and d have visited X, c (d) didn't visit Y** (non-contradiction)

    np.random.shuffle(template1)
    all_sentences_7 = []
    for i in range(int(upper_bound)):
        person1, place1 = template1[i]
        person2 = person1
        place2 = place1
        while person2 == person1 and place2 == place1:
            new_i = np.random.choice(len(template1))
            person2, place2 = template1[new_i]
        if i%2 ==0:
            person3 = person1
        else:
            person3 = person2
        if i%3 == 0:
            sentence = "{} and {} have traveled to {},{} didn't travel to {}".format(person1, person2, place1, person3, place2)
        elif i%3 ==1:
            sentence = "{} and {} went to {},{} didn't go to {}".format(person1, person2, place1, person3, place2)
        else:
            sentence = "{} and {} have visited {},{} didn't visit {}".format(person1, person2, place1, person3, place2)
        all_sentences_7.append(sentence)

    all_sentences_7 = [sentence.split(",") + [0] for sentence in all_sentences_7]

    # - 8) **c and d have traveled to X, e didn't travel to X** (non-contradiction)
    # - 8) **c and d went to X, e didn't go to X** (non-contradiction)
    # - 8) **c and d have visited X, e didn't visit X** (non-contradiction)

    np.random.shuffle(template1)
    all_sentences_8 = []
    for i in range(int(upper_bound)):
        person1, place1 = template1[i]
        person2 = person1
        while person2 == person1:
            new_i = np.random.choice(len(template1))
            person2, _ = template1[new_i]
        person3 = person1
        while person3 == person1 or person3 == person2:
            new_i = np.random.choice(len(template1))
            person3, _ = template1[new_i]
        if i % 3 == 0:
            sentence = "{} and {} have traveled to {},{} didn't travel to {}".format(person1, person2, place1, person3, place1)
        elif i % 3 == 1:
            sentence = "{} and {} went to {},{} didn't go to {}".format(person1, person2, place1, person3, place1)
        else:
            sentence = "{} and {} have visited {},{} didn't visit {}".format(person1, person2, place1, person3, place1)
        all_sentences_8.append(sentence)

    all_sentences_8 = [sentence.split(",") + [0] for sentence in all_sentences_8]


    np.random.shuffle(all_sentences_1)
    np.random.shuffle(all_sentences_2)
    np.random.shuffle(all_sentences_3)
    np.random.shuffle(all_sentences_4)
    np.random.shuffle(all_sentences_5)
    np.random.shuffle(all_sentences_6)
    np.random.shuffle(all_sentences_7)
    np.random.shuffle(all_sentences_8)


    size1 = len(all_sentences_1)
    size2 = len(all_sentences_2)
    size3 = len(all_sentences_3)
    size4 = len(all_sentences_4)
    size5 = len(all_sentences_5)
    size6 = len(all_sentences_6)
    size7 = len(all_sentences_7)
    size8 = len(all_sentences_8)

    all_sentences = all_sentences_1 + all_sentences_2 + all_sentences_3 + all_sentences_4
    all_sentences += all_sentences_5 + all_sentences_6 + all_sentences_7 + all_sentences_8




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


    # ### Saving to CSV

    if not os.path.exists("./data"):
        os.makedirs("data/")

    df_train.to_csv("data/boolean3_train.csv", index=False)
    df_test.to_csv("data/boolean3_test.csv", index=False)

if __name__ == '__main__':
    boolean3()