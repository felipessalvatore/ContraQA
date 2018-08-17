
# coding: utf-8

# # Dataset boolean6: implicit conjunction - different predicates applied on a person (reference problem)
# 
# Generating sentences of the form
# 
# - 1) **c, who is a PROFESSION, saw X, c didn't saw X** -- contradiction 
# - 1) **c, who is a PROFESSION, visit d, c didn't  visit d ** -- contradiction
# - 1) **c, who is a PROFESSION, bought Y, c didn't  buy Y ** -- contradiction 
# 
# 
# - 2) **c, who is a PROFESSION, saw X, c isn't a PROFESSION ** -- contradiction
# - 2) **c, who is a PROFESSION, visit d, c isn't a PROFESSION ** -- contradiction
# - 2) **c, who is a PROFESSION, bought Y, c isn't a PROFESSION ** -- contradiction
# 
# 
# - 3) **c, who is a PROFESSION1, saw X, c, who is a PROFESSION2, didn't saw X**  -- non-contradiction
# - 3) **c, who is a PROFESSION1, visit d, c, who is a PROFESSION2, didn't visit d**  -- non-contradiction
# - 3) **c, who is a PROFESSION1, bought Y, c, who is a PROFESSION2, didn't bought Y**  -- non-contradiction
# 
# 
# - 4) **c, who is a PROFESSION, saw X, c, e's father (mother), didn't saw X** -- non-contradiction
# - 4) **c, who is a PROFESSION, visit d, c, e's father (mother), didn't visit d** -- non-contradiction
# - 4) **c, who is a PROFESSION, bought Y, c, e's father (mother), didn't bought Y** -- non-contradiction
# 
# 
# - 5) **c saw Z running COMPLEMENT, c didn't saw Z running** -- contradiction 
# - 5) **c saw Z driving COMPLEMENT, c didn't saw Z driving** -- contradiction 
# 
# 
# 
# - 6) **c saw X running COMPLEMENT1, c didn't saw X running  COMPLEMENT2** -- non-contradiction
# - 6) **c saw X driving COMPLEMENT1, c didn't saw X driving COMPLEMENT2** -- non-contradiction 
# 
# 
# X = [a girl running, a blue plane, the new Tesla sport car, the new Marvel movie, a dog chasing a cat, etc.]
# 
# Y = [the new Tesla Roadster, etc.]
# 
# Z = [name, a girl, a boy]
# 
# 


import numpy as np
import pandas as pd
from word_lists import male_names, female_names, name_list
from word_lists import professions, color_list, city_list
import os

def get_new_item(item_list, src_list):
    size = len(src_list)
    new_item = src_list[np.random.choice(size)]
    while new_item in item_list: 
        new_i = np.random.choice(size)
        new_item = src_list[new_i]
    return new_item

def boolean6():

    upper_bound = 11000 / 6

    COMPLEMENT_runnig = ["from a parade car",
                         "from PERSON's car",
                          "from a COLOR car",
                          "from a COLOR bus",
                          "from a special recently built bus",
                          "from a COLOR school bus",
                          "from a bus going to CITY",
                          "from a COLOR bus with a new engine"]

    COMPLEMENT_driving = ["a COLOR car",
                          "in CITY",
                          "PERSON's car",
                          "a COLOR bicycle",
                          "PERSON's bicycle",
                          "an electric COLOR bicycle",
                          "PERSON's new Tesla Roadster"]

    what_I_see = ["a girl running",
                  "a blue plane",
                  "the new Tesla Roadster",
                  "the new Marvel movie from the Russo Brothers", 
                  "a dog chasing a cat",
                  "the accident",
                  "the car crash",
                  "a strange add",
                  "the city mayor"]

    what_I_buy = ["the new Tesla Roadster",
                  "the new Marvel movie from the Russo Brothers",
                  "a Chilean wine",
                  "a Macbook",
                  "a cup of coffee",
                  "a pizza",
                  "a japanese novel"]


    # - 1) **c, who is a PROFESSION, saw X, c didn't saw X** -- contradiction 
    # - 1) **c, who is a PROFESSION, visit d, c didn't  visit d ** -- contradiction
    # - 1) **c, who is a PROFESSION, bought Y, c didn't  buy Y ** -- contradiction 

    all_sentences_1 = []
    for i in range(int(upper_bound)):
        person1 = get_new_item([], name_list)
        person2 = get_new_item([person1], name_list)
        buy = get_new_item([], what_I_buy)
        see = get_new_item([], what_I_see)
        profession = get_new_item([], professions)
        if i % 3 == 0:
            sentence = "{}, who is a {}, saw {};{} didn't saw {}".format(person1, profession, see, person1, see)
        elif i % 3 ==1:
            sentence = "{}, who is a {}, visit {};{} didn't visit {}".format(person1, profession, person2, person1, person2)
        else:
            sentence = "{}, who is a {}, bought {};{} didn't buy {}".format(person1, profession, buy, person1, buy)
        all_sentences_1.append(sentence)

    all_sentences_1 = [sentence.split(";") + [1] for sentence in all_sentences_1]

    # - 2) **c, who is a PROFESSION, saw X, c isn't a PROFESSION ** -- contradiction
    # - 2) **c, who is a PROFESSION, visit d, c isn't a PROFESSION ** -- contradiction
    # - 2) **c, who is a PROFESSION, bought Y, c isn't a PROFESSION ** -- contradiction

    all_sentences_2 = []
    for i in range(int(upper_bound)):
        person1 = get_new_item([], name_list)
        person2 = get_new_item([person1], name_list)
        buy = get_new_item([], what_I_buy)
        see = get_new_item([], what_I_see)
        profession = get_new_item([], professions)
        if i % 3 == 0:
            sentence = "{}, who is a {}, saw {};{} isn't a {}".format(person1, profession, see, person1, profession)
        elif i % 3 ==1:
            sentence = "{}, who is a {}, visit {};{} isn't a {}".format(person1, profession, person2, person1, profession)
        else:
            sentence = "{}, who is a {}, bought {};{} isn't a {}".format(person1, profession, buy, person1, profession)
        all_sentences_2.append(sentence)

    all_sentences_2 = [sentence.split(";") + [1] for sentence in all_sentences_2]

    # - 3) **c, who is a PROFESSION1, saw X, c, who is a PROFESSION2, didn't saw X**  -- non-contradiction
    # - 3) **c, who is a PROFESSION1, visit d, c, who is a PROFESSION2, didn't visit d**  -- non-contradiction
    # - 3) **c, who is a PROFESSION1, bought Y, c, who is a PROFESSION2, didn't bought Y**  -- non-contradiction

    all_sentences_3 = []
    for i in range(int(upper_bound)):
        person1 = get_new_item([], name_list)
        person2 = get_new_item([person1], name_list)
        buy = get_new_item([], what_I_buy)
        see = get_new_item([], what_I_see)
        profession1 = get_new_item([], professions)
        profession2 = get_new_item([profession1], professions)
        assert profession1 != profession2 
        if i % 3 == 0:
            sentence = "{}, who is a {}, saw {};{}, who is a {}, didn't saw {}".format(person1, profession1, see, person1, profession2, see)
        elif i % 3 ==1:
            sentence = "{}, who is a {}, visit {};{}, who is a {}, didn't visit {}".format(person1, profession1, person2, person1, profession2, person2)
        else:
            sentence = "{}, who is a {}, bought {};{}, who is a {}, didn't buy {}".format(person1, profession1, buy, person1, profession2, buy)
        all_sentences_3.append(sentence)

    all_sentences_3 = [sentence.split(";") + [0] for sentence in all_sentences_3]

    # - 4) **c, who is a PROFESSION, saw X, c, e's father (mother), didn't saw X** -- non-contradiction
    # - 4) **c, who is a PROFESSION, visit d, c, e's father (mother), didn't visit d** -- non-contradiction
    # - 4) **c, who is a PROFESSION, bought Y, c, e's father (mother), didn't bought Y** -- non-contradiction

    all_sentences_4 = []
    for i in range(int(upper_bound)):
        if i % 2 == 0:
            person1 = get_new_item([], female_names)
            parent = "mother"
        else:
            person1 = get_new_item([], male_names)
            parent = "father"
        person2 = get_new_item([person1], name_list)
        person3 = get_new_item([person1, person2], name_list)
        buy = get_new_item([], what_I_buy)
        see = get_new_item([], what_I_see)
        profession1 = get_new_item([], professions)
        profession2 = get_new_item([profession1], professions)
        if i % 3 == 0:
            sentence = "{}, who is a {}, saw {};{}, {}'s {}, didn't saw {}".format(person1, profession1, see, person1, person3, parent, see)
        elif i % 3 ==1:
            sentence = "{}, who is a {}, visit {};{}, {}'s {}, didn't visit {}".format(person1, profession1, person2, person1,person3, parent,person2)
        else:
            sentence = "{}, who is a {}, bought {};{}, {}'s {}, didn't buy {}".format(person1, profession1, buy, person1, person3, parent, buy)
        all_sentences_4.append(sentence)

    all_sentences_4 = [sentence.split(";") + [0] for sentence in all_sentences_4]

    # - 5) **c saw X running COMPLEMENT, c didn't saw X running** -- contradiction 
    # - 5) **c saw X driving COMPLEMENT, c didn't saw X driving** -- contradiction 

    all_sentences_5 = []
    for i in range(int(upper_bound + 1)):
        person1 = get_new_item([], name_list)
        object_list = [get_new_item([person1], name_list), "a girl", "a boy"]
        direct_object = get_new_item([], object_list)
        if i % 2 == 0:
            verb = "running"
            complement = get_new_item([], COMPLEMENT_runnig)
        else:
            verb = "driving"
            complement = get_new_item([], COMPLEMENT_driving)
        person2 = get_new_item([person1, direct_object], name_list)
        color = get_new_item([], color_list)
        city = get_new_item([], city_list)
        complement = complement.replace("PERSON's", person2 + "'s")
        complement = complement.replace("COLOR", color)
        complement = complement.replace("CITY", city)
        sentence = "{} saw {} {} {};{} didn't saw {} {}".format(person1,
                                                                 direct_object,
                                                                 verb,
                                                                 complement,
                                                                 person1,
                                                                 direct_object,
                                                                 verb)
        all_sentences_5.append(sentence)

    all_sentences_5 = [sentence.split(";") + [1] for sentence in all_sentences_5]

    # - 6) **c saw X running COMPLEMENT1, c didn't saw X running  COMPLEMENT2** -- non-contradiction
    # - 6) **c saw X driving COMPLEMENT1, c didn't saw X driving COMPLEMENT2** -- non-contradiction 

    all_sentences_6 = []
    for i in range(int(upper_bound + 1)):
        person1 = get_new_item([], name_list)
        object_list = [get_new_item([person1], name_list), "a girl", "a boy"]
        direct_object = get_new_item([], object_list)
        if i % 2 == 0:
            verb = "running"
            COMPLEMENT = COMPLEMENT_runnig 
        else:
            verb = "driving"
            COMPLEMENT = COMPLEMENT_driving 
        
        complement1 = get_new_item([], COMPLEMENT)
        complement2 = get_new_item([complement1], COMPLEMENT)
        person2 = get_new_item([person1, direct_object], name_list)
        color = get_new_item([], color_list)
        city = get_new_item([], city_list)
        complement1 = complement1.replace("PERSON's", person2 + "'s")
        complement1 = complement1.replace("COLOR", color)
        complement1 = complement1.replace("CITY", city)
        person2 = get_new_item([person1, direct_object, person2], name_list)
        color = get_new_item([], color_list)
        city = get_new_item([], city_list)
        complement2 = complement2.replace("PERSON's", person2 + "'s")
        complement2 = complement2.replace("COLOR", color)
        complement2 = complement2.replace("CITY", city)
        sentence = "{} saw {} {} {};{} didn't saw {} {} {}".format(person1,
                                                                 direct_object,
                                                                 verb,
                                                                 complement1,
                                                                 person1,
                                                                 direct_object,
                                                                 verb,
                                                                 complement2)
        all_sentences_6.append(sentence)
        
    all_sentences_6 = [sentence.split(";") + [0] for sentence in all_sentences_6]

    np.random.shuffle(all_sentences_1)
    np.random.shuffle(all_sentences_2)
    np.random.shuffle(all_sentences_3)
    np.random.shuffle(all_sentences_4)
    np.random.shuffle(all_sentences_5)
    np.random.shuffle(all_sentences_6)


    size1 = len(all_sentences_1)
    size2 = len(all_sentences_2)
    size3 = len(all_sentences_3)
    size4 = len(all_sentences_4)
    size5 = len(all_sentences_5)
    size6 = len(all_sentences_6)


    all_sentences = all_sentences_1 + all_sentences_2 + all_sentences_3 + all_sentences_4
    all_sentences += all_sentences_5 + all_sentences_6

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

    df_train.to_csv("data/boolean6_train.csv", index=False)
    df_test.to_csv("data/boolean6_test.csv", index=False)

if __name__ == '__main__':
    boolean6()