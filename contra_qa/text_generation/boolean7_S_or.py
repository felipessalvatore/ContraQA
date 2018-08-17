
# coding: utf-8

# # Dataset boolean7: sentences conjoined by or
# 
# Generating sentences of the form
# 
# - 1) **c is P or d is Q, neither c is P nor d is Q** (contradiction)
# 
# - 2) **c is P or d is Q, c is not P and d is not Q** (contradiction)
# 
# - 3) **c is P or d is Q, c is not P or d is not Q** (non-contradiction)
# 
# - 4) **c is P or d is Q, c (d) is not P (Q)** (non-contradiction)

# In[1]:


import numpy as np
import pandas as pd
from word_lists import name_list, positive_personality_list
from word_lists import apparance_list, negative_personality_list
import os


def get_new_item(item_list, src_list):
    size = len(src_list)
    new_item = src_list[np.random.choice(size)]
    while new_item in item_list: 
        new_i = np.random.choice(size)
        new_item = src_list[new_i]
    return new_item


qualities = positive_personality_list + apparance_list + negative_personality_list
upper_bound = 11000/4


# ### Generating all types of sentences

def boolean7():

    # - 1) **c is P or d is Q, neither c is P nor d is Q** (contradiction)

    all_sentences_1 = []
    for i in range(int(upper_bound)):
        person1 = get_new_item([], name_list)
        person2 = get_new_item([person1], name_list)
        pred1 = get_new_item([], qualities)
        pred2 = get_new_item([pred1], qualities)
        sentence = "{} is {} or {} is {}, neither {} is {} nor {} is {}".format(person1,
                                                                                pred1,
                                                                                person2,
                                                                                pred2,
                                                                                person1,
                                                                                pred1,
                                                                                person2,
                                                                                pred2)
        all_sentences_1.append(sentence)
        
    all_sentences_1 = [sentence.split(",") + [1] for sentence in all_sentences_1]

    # - 2) **c is P or d is Q, c is not P and d is not Q** (contradiction)

    all_sentences_2 = []
    for i in range(int(upper_bound)):
        person1 = get_new_item([], name_list)
        person2 = get_new_item([person1], name_list)
        pred1 = get_new_item([], qualities)
        pred2 = get_new_item([pred1], qualities)
        sentence = "{} is {} or {} is {}, {} is not {} and {} is not {}".format(person1,
                                                                                pred1,
                                                                                person2,
                                                                                pred2,
                                                                                person1,
                                                                                pred1,
                                                                                person2,
                                                                                pred2)
        all_sentences_2.append(sentence)
        
    all_sentences_2 = [sentence.split(",") + [1] for sentence in all_sentences_2]


    # - 3) **c is P or d is Q, c is not P or d is not Q** (non-contradiction)

    all_sentences_3 = []
    for i in range(int(upper_bound)):
        person1 = get_new_item([], name_list)
        person2 = get_new_item([person1], name_list)
        pred1 = get_new_item([], qualities)
        pred2 = get_new_item([pred1], qualities)
        sentence = "{} is {} or {} is {}, {} is not {} or {} is not {}".format(person1,
                                                                                pred1,
                                                                                person2,
                                                                                pred2,
                                                                                person1,
                                                                                pred1,
                                                                                person2,
                                                                                pred2)
        all_sentences_3.append(sentence)
        
    all_sentences_3 = [sentence.split(",") + [0] for sentence in all_sentences_3]


    # - 4) **c is P or d is Q, c (d) is not P (Q)** (non-contradiction)


    all_sentences_4 = []
    for i in range(int(upper_bound)):
        person1 = get_new_item([], name_list)
        person2 = get_new_item([person1], name_list)
        pred1 = get_new_item([], qualities)
        pred2 = get_new_item([pred1], qualities)
        if i % 2 == 0:
            person3 = person1
            pred3 = pred1
        else:
            person3 = person2
            pred3 = pred2
        sentence = "{} is {} or {} is {}, {} is not {}".format(person1,
                                                               pred1,
                                                               person2,
                                                               pred2,
                                                               person3,
                                                               pred3)
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

    # ### Generating a train DataFrame with 10000 examples and a test DataFrame with 1000 examples 


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

    df_train.to_csv("data/boolean7_train.csv", index=False)
    df_test.to_csv("data/boolean7_test.csv", index=False)

if __name__ == '__main__':
    boolean7()