# # Dataset boolean1: simple negation
# 
# Generating sentences of the form
# 
# - 1) **c is P, c is not P** (contradiction)
# 
# - 2) **c is not P, c is P** (contradiction)
# 
# - 3) **c is P, c is Q** (non-contradiction)
# 
# - 4) **c is P, c is not Q** (non-contradiction)
# 
# - 5) **c is not P, c is Q** (non-contradiction)
# 
# - 6) **c is P, d is Q** (non-contradiction)
# 
# - 7) **c is P, d is not Q** (non-contradiction)
# 
# - 8) **c is not P, d is Q** (non-contradiction)
# 
# - 9) **c is P, d is not P** (non-contradiction)
# 
# - 10) **c is not P, d is P** (non-contradiction)
# 

import numpy as np
import pandas as pd
from word_lists import name_list, positive_personality_list, condition_list
import os
import itertools

def boolean1():
    template1 = itertools.product(name_list, positive_personality_list)
    template1 = list(template1)
    all_sentences_1 = ["{} is {}, {} is not {}".format(car,cdr,car,cdr) for car, cdr in template1]
    all_sentences_1 = [ sentence.split(",") + [1] for sentence in all_sentences_1]
    np.random.shuffle(all_sentences_1)


    template2 = itertools.product(name_list, condition_list)
    template2 = list(template2)
    all_sentences_2 = ["{} is not {}, {} is {}".format(car,cdr,car,cdr) for car, cdr in template2]
    all_sentences_2 = [ sentence.split(",") + [1] for sentence in all_sentences_2]
    np.random.shuffle(all_sentences_2)


    template3 = itertools.product(name_list, positive_personality_list, condition_list)
    template3 = list(template3)
    all_sentences_3 = ["{} is {}, {} is {}".format(car,
                                                   cdr,
                                                   car,
                                                   cddr) for car, cdr, cddr in template3]

    all_sentences_3 = [ sentence.split(",") + [0] for sentence in all_sentences_3]
    np.random.shuffle(all_sentences_3)



    all_sentences_4 = ["{} is {}, {} is not {}".format(car,cdr,car,cddr) for car, cdr, cddr in template3]

    all_sentences_4 = [ sentence.split(",") + [0] for sentence in all_sentences_4]

    np.random.shuffle(all_sentences_4)


    all_sentences_5 = ["{} is not {}, {} is {}".format(car,cdr,car,cddr) for car, cdr, cddr in template3]

    all_sentences_5 = [ sentence.split(",") + [0] for sentence in all_sentences_5]

    np.random.shuffle(all_sentences_5)


    np.random.shuffle(template3)

    template6 = template3[0:1100]

    all_sentences_6 = []

    for triple in template6:
        new_name = triple[0]
        while new_name == triple[0]:
            i = np.random.choice(len(name_list))
            new_name = name_list[i]
        car, cdr, cddr = triple 
        all_sentences_6.append("{} is {}, {} is {}".format(car,cdr,new_name,cddr))

    all_sentences_6 = [ sentence.split(",") + [0] for sentence in all_sentences_6]    

    np.random.shuffle(all_sentences_6)


    np.random.shuffle(template3)

    template7 = template3[0:1100]

    all_sentences_7 = []

    for triple in template7:
        new_name = triple[0]
        while new_name == triple[0]:
            i = np.random.choice(len(name_list))
            new_name = name_list[i]
        car, cdr, cddr = triple 
        all_sentences_7.append("{} is {}, {} is not {}".format(car,cdr,new_name,cddr))

    all_sentences_7 = [ sentence.split(",") + [0] for sentence in all_sentences_7]    
        
    np.random.shuffle(all_sentences_7)

    np.random.shuffle(template3)

    template8 = template3[0:1100]

    all_sentences_8 = []

    for triple in template8:
        new_name = triple[0]
        while new_name == triple[0]:
            i = np.random.choice(len(name_list))
            new_name = name_list[i]
        car, cdr, cddr = triple 
        all_sentences_8.append("{} is not {}, {} is {}".format(car,cdr,new_name,cddr))

    all_sentences_8 = [ sentence.split(",") + [0] for sentence in all_sentences_8]    
        
    np.random.shuffle(all_sentences_8)


    np.random.shuffle(template1)

    template9 = template1[0:1100]

    all_sentences_9 = []

    for car,cdr in template9:
        new_name = car
        while new_name == car:
            i = np.random.choice(len(name_list))
            new_name = name_list[i] 
        all_sentences_9.append("{} is {}, {} is not {}".format(car,cdr,new_name,cdr))

    all_sentences_9 = [ sentence.split(",") + [0] for sentence in all_sentences_9]

    np.random.shuffle(all_sentences_9)

    np.random.shuffle(template1)

    template10 = template1[0:1100]

    all_sentences_10 = []

    for car,cdr in template10:
        new_name = car
        while new_name == car:
            i = np.random.choice(len(name_list))
            new_name = name_list[i] 
        all_sentences_10.append("{} is not {}, {} is {}".format(car,cdr,new_name,cdr))

    all_sentences_10 = [ sentence.split(",") + [0] for sentence in all_sentences_10]

    np.random.shuffle(all_sentences_10)


    np.random.shuffle(all_sentences_1)
    np.random.shuffle(all_sentences_2)
    np.random.shuffle(all_sentences_3)
    np.random.shuffle(all_sentences_4)
    np.random.shuffle(all_sentences_5)
    np.random.shuffle(all_sentences_6)
    np.random.shuffle(all_sentences_7)
    np.random.shuffle(all_sentences_8)
    np.random.shuffle(all_sentences_9)
    np.random.shuffle(all_sentences_10)

    all_sentences_1 = all_sentences_1[0:2750]
    all_sentences_2 = all_sentences_2[0:2750]
    all_sentences_3 = all_sentences_3[0:688]
    all_sentences_4 = all_sentences_4[0:688]
    all_sentences_5 = all_sentences_5[0:688]
    all_sentences_6 = all_sentences_6[0:688]
    all_sentences_7 = all_sentences_7[0:688]
    all_sentences_8 = all_sentences_8[0:688]
    all_sentences_9 = all_sentences_9[0:688]
    all_sentences_10 = all_sentences_10[0:688]



    size1 = len(all_sentences_1)
    size2 = len(all_sentences_2)
    size3 = len(all_sentences_3)
    size4 = len(all_sentences_4)
    size5 = len(all_sentences_5)
    size6 = len(all_sentences_6)
    size7 = len(all_sentences_7)
    size8 = len(all_sentences_8)
    size9 = len(all_sentences_9)
    size10 = len(all_sentences_10)


    all_sentences = all_sentences_1 + all_sentences_2 + all_sentences_3 + all_sentences_4 + all_sentences_5
    all_sentences = all_sentences + all_sentences_6 + all_sentences_7 + all_sentences_8 + all_sentences_9 + all_sentences_10


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

    df_train.to_csv("data/boolean1_train.csv", index=False)
    df_test.to_csv("data/boolean1_test.csv", index=False)

if __name__ == '__main__':
    boolean1()