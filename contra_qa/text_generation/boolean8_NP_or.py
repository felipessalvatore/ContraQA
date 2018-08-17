
# coding: utf-8

# # Dataset boolean8: NP conjoined by or
# 
# Generating sentences of the form
# 
# - 1) **c traveled to X or Y, c traveled to neither X nor Y** (contradiction)
# - 1) **c went to X or Y, c went to neither X nor Y** (contradiction)
# - 1) **c visited X or Y, c visited neither X nor Y** (contradiction)
# 
# 
# - 2) **c traveled to X or Y, c didn't travel to X and c didn't travel to Y** (contradiction)
# - 2) **c went to X or Y, c didn't go to X and c didn't go to Y** (contradiction)
# - 2) **c visited X or Y, c didn't visit X and c didn't visit Y** (contradiction)
# 
# 
# - 3) **c traveled to X or Y, d(c) traveled to neither X(W) nor Y(W)** (non-contradiction)
# - 3) **c went to X or Y,  d(c) went to neither X(W) nor Y(W)** (non-contradiction)
# - 3) **c visited X or Y, d(c) visited neither X(W) nor Y(w)** (non-contradiction)
# 
# 
# - 4) **c traveled to X or Y, c didn't travel to X (Y)** (non-contradiction)
# - 4) **c went to X or Y, c didn't go to X(Y)** (non-contradiction)
# - 4) **c visited X or Y, c didn't visit X(Y)** (non-contradiction)
# 
# 
# - 5) **c or d got to the quarter finals last year, neither c nor d got to the quarter finals last year** (contradiction)
# - 5) **c or d won the last world cup, neither c nor d won the last world cup** (contradiction)
# - 5) **c or d are in the geopolitical position of e, neither c nor d are in the geopolitical position of e** (contradiction)
# 
# 
# - 6) **c or d got to the quarter finals last year, c didn't get to the quarter finals last year and d didn't get to the quarter finals last year** (contradiction)
# - 6) **c or d won the last world cup, c didn't win the last world cup and d didn't win the last world cup** (contradiction)
# - 6) **c or d are in the geopolitical position of e, c isn't in the geopolitical position of e and d isn't in the geopolitical position of e ** (contradiction)
# 
# 
# - 7) **c or d got to the quarter finals last year, neither c(e) nor d(e) got to the quarter finals last year** (non-contradiction)
# - 7) **c or d won the last world cup, neither c(e) nor d(e) won the last world cup** (non-contradiction)
# - 7) **c or d are in the geopolitical position of e, neither c(f) nor d(f) are in the geopolitical position of e** (non-contradiction)
# 
# 
# - 8) **c or d got to the quarter finals last year, c (d) didn't get to the quarter finals last year** (non-contradiction)
# - 8) **c or d won the last world cup, neither c (d) didn't winthe last world cup ** (non-contradiction)
# - 8) **c or d are in the geopolitical position of e, c (d) isn't in the geopolitical position of e** (non-contradiction)
# 

# In[1]:


import numpy as np
import pandas as pd
from word_lists import name_list, city_list, team_list
import os

def get_new_item(item_list, src_list):
    size = len(src_list)
    new_item = src_list[np.random.choice(size)]
    while new_item in item_list: 
        new_i = np.random.choice(size)
        new_item = src_list[new_i]
    return new_item

upper_bound = 11000 / 8


# ### Generating all types of sentences

def boolean8():

    # - 1) **c traveled to X or Y, c traveled to neither X nor Y** (contradiction)
    # - 1) **c went to X or Y,  c went to neither X nor Y** (contradiction)
    # - 1) **c visited X or Y, c visited neither X nor Y** (contradiction)

    all_sentences_1 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        city1 = get_new_item([], city_list)
        city2 = get_new_item([city1], city_list)
        if i % 3 == 0:
            sentence = "{} traveled to {} or {},{} traveled to neither {} nor {}".format(person, city1, city2, person, city1, city2)
        elif i % 3 == 1:
            sentence = "{} went to {} or {},{} went to neither {} nor {}".format(person, city1, city2, person, city1, city2)
        else:
            sentence = "{} has visited {} or {},{} visited neither {} nor {}".format(person, city1, city2, person, city1, city2)
        all_sentences_1.append(sentence)

        
    all_sentences_1 = [sentence.split(",") + [1] for sentence in all_sentences_1]

    # - 2) **c traveled to X or Y, c didn't travel to X and c didn't travel to Y** (contradiction)
    # - 2) **c went to X or Y, c didn't go to X and c didn't go to Y** (contradiction)
    # - 2) **c visited X or Y, c didn't visit X and c didn't visit Y** (contradiction)

    all_sentences_2 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        city1 = get_new_item([], city_list)
        city2 = get_new_item([city1], city_list)
        if i % 3 == 0:
            sentence = "{} traveled to {} or {},{} didn't travel to {} and {} didn't travel to {}".format(person,
                                                                                                          city1,
                                                                                                          city2,
                                                                                                          person,
                                                                                                          city1,
                                                                                                          person,
                                                                                                          city2)
        elif i % 3 == 1:
            sentence = "{} went to {} or {},{} didn't go to {} and {} didn't go to {}".format(person,
                                                                                              city1,
                                                                                              city2,
                                                                                              person,
                                                                                              city1,
                                                                                              person,
                                                                                              city2)
        else:
            sentence = "{} has visited {} or {},{} didn't visit {} and {} didn't visit {}".format(person,
                                                                                                  city1,
                                                                                                  city2,
                                                                                                  person,
                                                                                                  city1,
                                                                                                  person,
                                                                                                  city2)
        
        all_sentences_2.append(sentence)

        
    all_sentences_2 = [sentence.split(",") + [1] for sentence in all_sentences_2]

    # - 3) **c traveled to X or Y, d(c) traveled to neither X(W) nor Y(W)** (non-contradiction)
    # - 3) **c went to X or Y,  d(c) went to neither X(W) nor Y(W)** (non-contradiction)
    # - 3) **c visited X or Y, d(c) visited neither X(W) nor Y(w)** (non-contradiction)

    all_sentences_3 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        other_person = get_new_item([person], name_list)
        city1 = get_new_item([], city_list)
        city2 = get_new_item([city1], city_list)
        city3 = get_new_item([city1, city2], city_list)
        
        if i % 2 == 0:
            person_p = other_person
            city_p_1 = city1
            city_p_2 = city2
        else:
            person_p = person
            if i % 3:
                city_p_1 = city3
                city_p_2 = city2
            else:
                city_p_1 = city1
                city_p_2 = city3
        
        if i % 3 == 0:
            sentence = "{} traveled to {} or {},{} traveled to neither {} nor {}".format(person,
                                                                                         city1,
                                                                                         city2,
                                                                                         person_p,
                                                                                         city_p_1,
                                                                                         city_p_2)
        elif i % 3 == 1:
            sentence = "{} went to {} or {},{} went to neither {} nor {}".format(person,
                                                                                 city1,
                                                                                 city2,
                                                                                 person_p,
                                                                                 city_p_1,
                                                                                 city_p_2)
        else:
            sentence = "{} has visited {} or {},{} visited neither {} nor {}".format(person,
                                                                                     city1,
                                                                                     city2,
                                                                                     person_p,
                                                                                     city_p_1,
                                                                                     city_p_2)        
            
        all_sentences_3.append(sentence)

        
    all_sentences_3 = [sentence.split(",") + [0] for sentence in all_sentences_3]

    # - 4) **c traveled to X or Y, c didn't travel to X (Y)** (non-contradiction)
    # - 4) **c went to X or Y, c didn't go to X(Y)** (non-contradiction)
    # - 4) **c visited X or Y, c didn't visit X(Y)** (non-contradiction)

    all_sentences_4 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        city1 = get_new_item([], city_list)
        city2 = get_new_item([city1], city_list)
        
        if i % 2 == 0:
            city3 = city1
        else:
            city3 = city2

        if i % 3 == 0:
            sentence = "{} traveled to {} or {},{} didn't travel to {}".format(person,
                                                                               city1,
                                                                               city2,
                                                                               person,
                                                                               city3)
        elif i % 3 == 1:
            sentence = "{} went to {} or {},{} didn't go to {}".format(person,
                                                                       city1,
                                                                       city2,
                                                                       person,
                                                                       city3)
        else:
            sentence = "{} has visited {} or {},{} didn't visit {}".format(person,
                                                                           city1,
                                                                           city2,
                                                                           person,
                                                                           city3)
        
        all_sentences_4.append(sentence)

        
    all_sentences_4 = [sentence.split(",") + [0] for sentence in all_sentences_4]

    # - 5) **c or d got to the quarter finals last year, neither c nor d got to the quarter finals last year** (contradiction)
    # - 5) **c or d won the last world cup, neither c nor d won the last world cup** (contradiction)
    # - 5) **c or d are in the geopolitical position of e, neither c nor d are in the geopolitical position of e** (contradiction)

    all_sentences_5 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        team1 = get_new_item([], team_list)
        team2 = get_new_item([team1], team_list)
        team3 = get_new_item([team1, team2], team_list)
        if i % 3 == 0:
            sentence = "{} or {} got to the quarter finals last year, neither {} nor {} got to the quarter finals last year".format(team1,
                                                                                                                                    team2,
                                                                                                                                    team1,
                                                                                                                                    team2)
        elif i % 3 == 1:
            sentence = "{} or {} won the last world cup, neither {} nor {} won the last world cup".format(team1,
                                                                                                          team2,
                                                                                                          team1,
                                                                                                          team2)    
        else:
            sentence = "{} or {} are in the geopolitical position of {}, neither {} nor {} are in the geopolitical position of {}".format(team1,
                                                                                                                                          team2,
                                                                                                                                          team3,
                                                                                                                                          team1,
                                                                                                                                          team2,
                                                                                                                                          team3)

            
        all_sentences_5.append(sentence)

        
    all_sentences_5 = [sentence.split(",") + [1] for sentence in all_sentences_5]

    # - 6) **c or d got to the quarter finals last year, c didn't get to the quarter finals last year and d didn't get to the quarter finals last year** (contradiction)
    # - 6) **c or d won the last world cup, c didn't win the last world cup and d didn't win the last world cup** (contradiction)
    # - 6) **c or d are in the geopolitical position of e, c isn't in the geopolitical position of e and d isn't in the geopolitical position of e ** (contradiction)

    all_sentences_6 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        team1 = get_new_item([], team_list)
        team2 = get_new_item([team1], team_list)
        team3 = get_new_item([team1, team2], team_list)
        if i % 3 == 0:
            sentence = "{} or {} got to the quarter finals last year,{} didn't get to the quarter finals last year and {} didn't get to the quarter finals last year".format(team1,
                                                                                                                                    team2,
                                                                                                                                    team1,
                                                                                                                                    team2)
        elif i % 3 == 1:
            sentence = "{} or {} won the last world cup,{} didn't win the last world cup and {} didn't win the last world cup".format(team1,
                                                                                                                                      team2,
                                                                                                                                      team1,
                                                                                                                                      team2)    
        else:
            sentence = "{} or {} are in the geopolitical position of {},{} isn't in the geopolitical position of {} and {} isn't in the geopolitical position of {}".format(team1,
                                                                                                                                                                            team2,
                                                                                                                                                                            team3,
                                                                                                                                                                            team1,
                                                                                                                                                                            team3,
                                                                                                                                                                            team2,
                                                                                                                                                                            team3)

            
        all_sentences_6.append(sentence)

        
    all_sentences_6 = [sentence.split(",") + [1] for sentence in all_sentences_6]

    # - 7) **c or d got to the quarter finals last year, neither c(e) nor d(e) got to the quarter finals last year** (non-contradiction)
    # - 7) **c or d won the last world cup, neither c(e) nor d(e) won the last world cup** (non-contradiction)
    # - 7) **c or d are in the geopolitical position of e, neither c(f) nor d(f) are in the geopolitical position of e** (non-contradiction)

    all_sentences_7 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        team1 = get_new_item([], team_list)
        team2 = get_new_item([team1], team_list)
        team3 = get_new_item([team1, team2], team_list)
        team4 = get_new_item([team1, team2, team3], team_list)
        
        if i % 2 == 0:
            team_p_1 = team1 
            team_p_2 = team4
        else:
            team_p_1 = team4
            team_p_2 = team1
        
        if i % 3 == 0:
            sentence = "{} or {} got to the quarter finals last year, neither {} nor {} got to the quarter finals last year".format(team1,
                                                                                                                                    team2,
                                                                                                                                    team_p_1,
                                                                                                                                    team_p_2)
        elif i % 3 == 1:
            sentence = "{} or {} won the last world cup, neither {} nor {} won the last world cup".format(team1,
                                                                                                          team2,
                                                                                                          team_p_1,
                                                                                                          team_p_2)    
        else:
            sentence = "{} or {} are in the geopolitical position of {}, neither {} nor {} are in the geopolitical position of {}".format(team1,
                                                                                                                                          team2,
                                                                                                                                          team3,
                                                                                                                                          team_p_1,
                                                                                                                                          team_p_2,
                                                                                                                                          team3)

            
        all_sentences_7.append(sentence)

        
    all_sentences_7 = [sentence.split(",") + [0] for sentence in all_sentences_7]

    all_sentences_8 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        team1 = get_new_item([], team_list)
        team2 = get_new_item([team1], team_list)
        team3 = get_new_item([team1, team2], team_list)
        if i % 2 == 0:
            team_p = team1 
        else:
            team_p = team2
            
        if i % 3 == 0:
            sentence = "{} or {} got to the quarter finals last year,{} didn't get to the quarter finals last year".format(team1,
                                                                                                                           team2,
                                                                                                                           team_p)
        elif i % 3 == 1:
            sentence = "{} or {} won the last world cup,{} didn't win the last world cup".format(team1,
                                                                                                                                      team2,
                                                                                                                                      team_p)    
        else:
            sentence = "{} or {} are in the geopolitical position of {},{} isn't in the geopolitical position of {}".format(team1,
                                                                                                                            team2,
                                                                                                                            team3,
                                                                                                                            team_p,
                                                                                                                            team3)

            
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

    df_train.to_csv("data/boolean8_train.csv", index=False)
    df_test.to_csv("data/boolean8_test.csv", index=False)

if __name__ == '__main__':
    boolean8()