
# coding: utf-8

# # Dataset boolean9: VP conjoined by and
# 
# Generating sentences of the form
# 
# - 1) **c will VERB1 COMPLEMENT1 or  VERB2 COMPLEMENT2, c will neither VERB1 COMPLEMENT1 nor VERB2 COMPLEMENT2 ** (contradiction)
# 
# 
# - 2) **c will VERB1 COMPLEMENT1 or VERB2 COMPLEMENT2, c will not VERB1 COMPLEMENT1 and c will not  VERB2 COMPLEMENT2 ** (contradiction)
# 
# 
# - 3) **c will VERB1 COMPLEMENT1 or VERB2 COMPLEMENT2, c will not VERB1 COMPLEMENT1 (VERB2 COMPLEMENT2)** (non-contradiction)
# 
# 
# - 4) **c will VERB1 COMPLEMENT1 or  VERB2 COMPLEMENT2, d(c) will neither VERB1 COMPLEMENT1 (VERB3 COMPLEMENT3) nor VERB2 COMPLEMENT2 (VERB3 COMPLEMENT3) ** (non-contradiction)
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


for i in range(len(name_list)):
    name = get_new_item([], name_list)
    new_name = get_new_item([name], name_list)
    assert name != new_name and name in name_list and new_name in name_list

def replace_words(input_str, person, color, city, profession, pred):
    input_str = input_str.replace("PERSON's", person + "'s")
    input_str = input_str.replace("PERSON", person)    
    input_str = input_str.replace("COLOR", color)
    input_str = input_str.replace("CITY", city)
    input_str = input_str.replace("PROFESSION", profession)
    input_str = input_str.replace("PRED", pred)
    return input_str

upper_bound = 11000 / 4


# Creating a dict of infinitv verbs 2 context

# #### IMPORTANTE EH PRECISO VER SE TODOS OS CONTEXTOS ESTAO NO FUTURO

inf2past = {}
for k, v in verbs_third_person_past_infinitiv_neg.items():
    if k != "was":
        words = v.split(" ")
        inf2past[words[1]] = k

infinitv_verbs = list(inf2past.keys())        

infinitiv_CONTEXT = {}
for verb in infinitv_verbs:
    infinitiv_CONTEXT[verb] = verbs_third_person_past_CONTEXT[inf2past[verb]]


# - 1) **c will VERB1 COMPLEMENT1 or  VERB2 COMPLEMENT2, c will neither VERB1 COMPLEMENT1 nor VERB2 COMPLEMENT2 ** (contradiction)

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
    verb1 = get_new_item([], infinitv_verbs)
    verb2 = get_new_item([verb1], infinitv_verbs)
    context = infinitiv_CONTEXT[verb1]
    complement1 = get_new_item([], context)
    complement1 = replace_words(complement1, person1, color1, city1, profession1, pred1)
    context = infinitiv_CONTEXT[verb2]
    complement2 = get_new_item([], context)
    complement2 = replace_words(complement2, person2, color2, city2, profession2, pred2)
    sentence = "{} will {} {} or {} {},{} will neither {} {} nor {} {}".format(person,
                                                                                  verb1,
                                                                                  complement1,
                                                                                  verb2,
                                                                                  complement2,
                                                                                  person,
                                                                                  verb1,
                                                                                  complement1,
                                                                                  verb2,
                                                                                  complement2,)
    all_sentences_1.append(sentence)

all_sentences_1 = [sentence.split(",") + [1] for sentence in all_sentences_1]

# - 2) **c will VERB1 COMPLEMENT1 or VERB2 COMPLEMENT2, c will not VERB1 COMPLEMENT1 and c will not  VERB2 COMPLEMENT2 ** (contradiction)

all_sentences_2 = []
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
    verb1 = get_new_item([], infinitv_verbs)
    verb2 = get_new_item([verb1], infinitv_verbs)
    context = infinitiv_CONTEXT[verb1]
    complement1 = get_new_item([], context)
    complement1 = replace_words(complement1, person1, color1, city1, profession1, pred1)
    context = infinitiv_CONTEXT[verb2]
    complement2 = get_new_item([], context)
    complement2 = replace_words(complement2, person2, color2, city2, profession2, pred2)
    sentence = "{} will {} {} or {} {},{} will not {} {} and {} wil not {} {}".format(person,
                                                                                  verb1,
                                                                                  complement1,
                                                                                  verb2,
                                                                                  complement2,
                                                                                  person,
                                                                                  verb1,
                                                                                  complement1,
                                                                                  person,
                                                                                  verb2,
                                                                                  complement2,)
    all_sentences_2.append(sentence)

all_sentences_2 = [sentence.split(",") + [1] for sentence in all_sentences_2]


# - 3) **c will VERB1 COMPLEMENT1 or VERB2 COMPLEMENT2, c will not VERB1 COMPLEMENT1 (VERB2 COMPLEMENT2)** (non-contradiction)

all_sentences_3 = []
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
    verb1 = get_new_item([], infinitv_verbs)
    verb2 = get_new_item([verb1], infinitv_verbs)
    context = infinitiv_CONTEXT[verb1]
    complement1 = get_new_item([], context)
    complement1 = replace_words(complement1, person1, color1, city1, profession1, pred1)
    context = infinitiv_CONTEXT[verb2]
    complement2 = get_new_item([], context)
    complement2 = replace_words(complement2, person2, color2, city2, profession2, pred2)
    if i % 2 == 0:
        verb_p = verb1
        complement_p = complement1
    else:
        verb_p = verb2
        complement_p = complement2
    sentence = "{} will {} {} or {} {},{} will not {} {}".format(person,
                                                                 verb1,
                                                                 complement1,
                                                                 verb2,
                                                                 complement2,
                                                                 person,
                                                                 verb_p,
                                                                 complement_p)
    all_sentences_3.append(sentence)

all_sentences_3 = [sentence.split(",") + [0] for sentence in all_sentences_3]


# - 4) **c will VERB1 COMPLEMENT1 or  VERB2 COMPLEMENT2, d(c) will neither VERB1 COMPLEMENT1 (VERB3 COMPLEMENT3) nor VERB2 COMPLEMENT2 (VERB3 COMPLEMENT3) ** (non-contradiction)

all_sentences_4 = []
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
    verb1 = get_new_item([], infinitv_verbs)
    verb2 = get_new_item([verb1], infinitv_verbs)
    verb3 = get_new_item([verb1, verb2], infinitv_verbs)
    context = infinitiv_CONTEXT[verb1]
    complement1 = get_new_item([], context)
    complement1 = replace_words(complement1, person1, color1, city1, profession1, pred1)
    context = infinitiv_CONTEXT[verb2]
    complement2 = get_new_item([], context)
    complement2 = replace_words(complement2, person2, color2, city2, profession2, pred2)
    context = infinitiv_CONTEXT[verb3]
    complement3 = get_new_item([], context)
    complement3 = replace_words(complement3, person3, color3, city3, profession3, pred3)
    if i % 2 == 0:
        person_p = other_person
        verb_p_1 =  verb1
        complement_p_1 =  complement1
        verb_p_2 =  verb2
        complement_p_2 =  complement2
    else:
        if i % 3 == 0:
            person_p = person
            verb_p_1 =  verb1
            complement_p_1 =  complement1
            verb_p_2 =  verb3
            complement_p_2 =  complement3
        else:
            person_p = person
            verb_p_1 =  verb3
            complement_p_1 =  complement3
            verb_p_2 =  verb2
            complement_p_2 =  complement2
        
    sentence = "{} will {} {} or {} {},{} will neither {} {} nor {} {}".format(person,
                                                                               verb1,
                                                                               complement1,
                                                                               verb2,
                                                                               complement2,
                                                                               person_p,
                                                                               verb_p_1,
                                                                               complement_p_1,
                                                                               verb_p_2,
                                                                               complement_p_2)
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

df_train.to_csv("data/boolean9_train.csv", index=False)
df_test.to_csv("data/boolean9_test.csv", index=False)
