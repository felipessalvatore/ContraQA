
# coding: utf-8

# # Dataset boolean4: VP conjoined by and
# 
# Generating sentences of the form
# 
# - 1) **c VERB1 COMPLEMENT1 AND  VERB2 COMPLEMENT2, c didn't VERB1 COMPLEMENT1** (contradiction)
# - 1) **c VERB1 COMPLEMENT1 AND  VERB2 COMPLEMENT2, c didn't VERB2 COMPLEMENT2** (contradiction)
# 
# 
# - 2) **c VERB1 COMPLEMENT1 AND  VERB2 COMPLEMENT2, c didn't VERB3 COMPLEMENT3** (non-contradiction)
# - 2) **c VERB1 COMPLEMENT1 AND  VERB2 COMPLEMENT2, d didn't VERB1 (VERB2) COMPLEMENT1 (COMPLEMENT1)** (non-contradiction)
# 

import numpy as np
import pandas as pd
from word_lists import name_list
from word_lists import professions, color_list, city_list
from word_lists import verbs_third_person_past, verbs_third_person_past_CONTEXT
from word_lists import verbs_third_person_past_infinitiv_neg, positive_personality_list
import os



def get_new_item(item_list, src_list):
    size = len(src_list)
    new_item = src_list[np.random.choice(size)]
    while new_item in item_list: 
        new_i = np.random.choice(size)
        new_item = src_list[new_i]
    return new_item


def replace_words(input_str, person, color, city, profession, pred):
    input_str = input_str.replace("PERSON's", person + "'s")
    input_str = input_str.replace("PERSON", person)    
    input_str = input_str.replace("COLOR", color)
    input_str = input_str.replace("CITY", city)
    input_str = input_str.replace("PROFESSION", profession)
    input_str = input_str.replace("PRED", pred)
    return input_str


def boolean4():

    upper_bound = 11000 / 2


    # - 1) **c VERB1 COMPLEMENT1 AND  VERB2 COMPLEMENT2, c didn't VERB1 COMPLEMENT1** (contradiction)
    # - 1) **c VERB1 COMPLEMENT1 AND  VERB2 COMPLEMENT2, c didn't VERB2 COMPLEMENT2** (contradiction)

    all_sentences_1 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        person1 = get_new_item([person], name_list)
        person2 = get_new_item([person, person1], name_list)
        pred1 = get_new_item([], positive_personality_list)
        pred2 = get_new_item([pred1], positive_personality_list)
        color1 = get_new_item([], color_list)
        color2 = get_new_item([color1], color_list)
        city1 =  get_new_item([], city_list)
        city2 =  get_new_item([city1], city_list)
        profession1 = get_new_item([], professions)
        profession2 = get_new_item([profession1], professions)
        verb1 = get_new_item([], verbs_third_person_past)
        verb2 = get_new_item([verb1], verbs_third_person_past)
        context = verbs_third_person_past_CONTEXT[verb1]
        complement1 = get_new_item([], context)
        complement1 = replace_words(complement1, person1, color1, city1, profession1, pred1)
        context = verbs_third_person_past_CONTEXT[verb2]
        complement2 = get_new_item([], context)
        complement2 = replace_words(complement2, person2, color2, city2, profession2, pred2)
        if i % 2 == 0:
            verb3 = verbs_third_person_past_infinitiv_neg[verb1]
            complement3 = complement1
        else:
            verb3 = verbs_third_person_past_infinitiv_neg[verb2]
            complement3 = complement2
        sentence = "{} {} {} and {} {},{} {} {}".format(person, verb1, complement1, verb2, complement2, person, verb3, complement3)
        all_sentences_1.append(sentence)

    all_sentences_1 = [sentence.split(",") + [1] for sentence in all_sentences_1]


    # - 2) **c VERB1 COMPLEMENT1 AND  VERB2 COMPLEMENT2, c didn't VERB3 COMPLEMENT3** (non-contradiction)
    # - 2) **c VERB1 COMPLEMENT1 AND  VERB2 COMPLEMENT2, d didn't VERB1 (VERB2) COMPLEMENT1 (COMPLEMENT1)** (non-contradiction)

    all_sentences_2 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        person1 = get_new_item([person], name_list)
        person2 = get_new_item([person, person1], name_list)
        person3 = get_new_item([person, person1, person2], name_list)
        other_person = get_new_item([person, person1, person2, person3], name_list)
        color1 = get_new_item([], color_list)
        color2 = get_new_item([color1], color_list)
        color3 = get_new_item([color1, color2], color_list)
        city1 =  get_new_item([], city_list)
        city2 =  get_new_item([city1], city_list)
        city3 = get_new_item([city1, city2], city_list)
        profession1 = get_new_item([], professions)
        profession2 = get_new_item([profession1], professions)
        profession3 = get_new_item([profession1, profession2], professions)
        pred1 = get_new_item([], positive_personality_list)
        pred2 = get_new_item([pred1], positive_personality_list)
        pred3 = get_new_item([pred1, pred2], positive_personality_list)
        verb1 = get_new_item([], verbs_third_person_past)
        verb2 = get_new_item([verb1], verbs_third_person_past)
        verb3 = get_new_item([verb1, verb2], verbs_third_person_past)
        context = verbs_third_person_past_CONTEXT[verb1]
        complement1 = get_new_item([], context)
        complement1 = replace_words(complement1, person1, color1, city1, profession1, pred1)
        context = verbs_third_person_past_CONTEXT[verb2]
        complement2 = get_new_item([], context)
        complement2 = replace_words(complement2, person2, color2, city2, profession2, pred2)
        context = verbs_third_person_past_CONTEXT[verb3]
        complement3 = get_new_item([], context)
        complement3 = replace_words(complement3, person3, color3, city3, profession3, pred3)
        if i % 3 == 0:
            new_person = person
            verb3 = verbs_third_person_past_infinitiv_neg[verb3]
        elif i % 3 == 1:
            new_person = other_person
            verb3 = verbs_third_person_past_infinitiv_neg[verb1]
            complement3 = complement1
            
        else:
            new_person = other_person
            verb3 = verbs_third_person_past_infinitiv_neg[verb2]
            complement3 = complement2
        sentence = "{} {} {} and {} {},{} {} {}".format(person, verb1, complement1, verb2, complement2, new_person, verb3, complement3)
        all_sentences_2.append(sentence)

    all_sentences_2 = [sentence.split(",") + [0] for sentence in all_sentences_2]

    np.random.shuffle(all_sentences_1)
    np.random.shuffle(all_sentences_2)


    size1 = len(all_sentences_1)
    size2 = len(all_sentences_2)


    all_sentences = all_sentences_1 + all_sentences_2
    size = len(all_sentences)

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

    # In[13]:


    if not os.path.exists("./data"):
        os.makedirs("data/")

    df_train.to_csv("data/boolean4_train.csv", index=False)
    df_test.to_csv("data/boolean4_test.csv", index=False)

if __name__ == '__main__':
    boolean4()