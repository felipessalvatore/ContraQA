
# coding: utf-8

# # Dataset boolean10: AP conjoined by or
# 
# Generating sentences of the form
# 
# - 1) **c's painting is P or Q, c's painting is neither P nor Q** (contradiction)
# 
# - 2) **c's painting is P or Q, c's painting isn't P and c's paiting isn't Q** (contradiction)
# 
# - 3) **c's painting is P or Q, c's painting isn't P (Q)** (non-contradiction)
# 
# - 4) **c's painting is P or Q, d's (c's) painting is neither P (W) nor Q (W)** (non-contradiction)
# 

# In[1]:


import numpy as np
import pandas as pd
from word_lists import name_list, all_attributes
import os

def get_new_item(item_list, src_list):
    size = len(src_list)
    new_item = src_list[np.random.choice(size)]
    while new_item in item_list: 
        new_i = np.random.choice(size)
        new_item = src_list[new_i]
    return new_item

def boolean10():
    upper_bound = 11000 / 4
    vowels = 'aeiou'


    # ### Generating all types of sentences

    # - 1) **c's painting is P or Q, c's painting is neither P nor Q** (contradiction)

    all_sentences_1 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        pred1 = get_new_item([], all_attributes)
        pred2 = get_new_item([pred1], all_attributes)
        sentence = "{}'s painting is {} or {}, {}'s painting is neither {} nor {}".format(person,
                                                                                          pred1,
                                                                                          pred2,
                                                                                          person,
                                                                                          pred1,
                                                                                          pred2)
        all_sentences_1.append(sentence)

    all_sentences_1 = [sentence.split(",") + [1] for sentence in all_sentences_1]

    # - 2) **c's painting is P or Q, c's painting isn't P and c's paiting isn't Q** (contradiction)

    all_sentences_2 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        pred1 = get_new_item([], all_attributes)
        pred2 = get_new_item([pred1], all_attributes)
        sentence = "{}'s painting is {} or {}, {}'s painting isn't {} and {}'s painting isn't {}".format(person,
                                                                                                         pred1,
                                                                                                         pred2,
                                                                                                         person,
                                                                                                         pred1,
                                                                                                         person,
                                                                                                         pred2)
        all_sentences_2.append(sentence)

    all_sentences_2 = [sentence.split(",") + [1] for sentence in all_sentences_2]

    # - 3) **c's painting is P or Q, c's painting isn't P (Q)** (non-contradiction)

    all_sentences_3 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        pred1 = get_new_item([], all_attributes)
        pred2 = get_new_item([pred1], all_attributes)
        if i % 2 == 0:
            pred_p = pred1
        else:
            pred_p = pred2
            
        sentence = "{}'s painting is {} or {}, {}'s painting isn't {}".format(person,
                                                                              pred1,
                                                                              pred2,
                                                                              person,
                                                                              pred_p)
        all_sentences_3.append(sentence)

    all_sentences_3 = [sentence.split(",") + [0] for sentence in all_sentences_3]

    # - 4) **c's painting is P or Q, d's (c's) painting is neither P (W) nor Q (W)** (non-contradiction)

    all_sentences_4 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        other_person = get_new_item([person], name_list) 
        pred1 = get_new_item([], all_attributes)
        pred2 = get_new_item([pred1], all_attributes)
        pred3 = get_new_item([pred1, pred2], all_attributes)
        
        if i % 2 == 0:
            person_p = other_person
            pred_p_1 = pred1
            pred_p_2 = pred2
        else:
            if i % 3 == 0:
                person_p = person
                pred_p_1 = pred1
                pred_p_2 = pred3
            else:
                person_p = person
                pred_p_1 = pred3
                pred_p_2 = pred2
        
        sentence = "{}'s painting is {} or {}, {}'s painting is neither {} nor {}".format(person,
                                                                                          pred1,
                                                                                          pred2,
                                                                                          person_p,
                                                                                          pred_p_1,
                                                                                          pred_p_2)
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

    df_train.to_csv("data/boolean10_train.csv", index=False)
    df_test.to_csv("data/boolean10_test.csv", index=False)

if __name__ == '__main__':
    boolean10()
