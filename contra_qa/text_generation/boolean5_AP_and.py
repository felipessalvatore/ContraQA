
# coding: utf-8

# # Dataset boolean5: AP conjoined by and
#
# Generating sentences of the form
#
# - 1) **c created a(n) P and Q work of art, c didn't create a(n) P (Q) work of art** (contradiction)
#
# - 2) **c created a(n) P and Q work of art, c (d) didn't create a(n) W (P, Q)  work of art** (non-contradiction)

import numpy as np
import pandas as pd
try:
    from word_lists import name_list, all_attributes
except Exception as e:
    from contra_qa.text_generation.word_lists import name_list, all_attributes
import os


def get_new_item(item_list, src_list):
    size = len(src_list)
    new_item = src_list[np.random.choice(size)]
    while new_item in item_list:
        new_i = np.random.choice(size)
        new_item = src_list[new_i]
    return new_item


def boolean5():

    upper_bound = 11000 / 2
    vowels = 'aeiou'

    # ### Generating all types of sentences

    # - 1) **c created a(n) P and Q work of art, c didn't create a(n) P (Q) work of art** (contradiction)

    all_sentences_1 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        pred1 = get_new_item([], all_attributes)
        pred2 = get_new_item([pred1], all_attributes)
        if pred1[0] in vowels:
            preposition1 = "an"
        else:
            preposition1 = "a"
        if pred2[0] in vowels:
            preposition2 = "an"
        else:
            preposition2 = "a"
        if i % 2 == 0:
            preposition3 = preposition1
            pred3 = pred1
        else:
            preposition3 = preposition1
            pred3 = pred1
        sentence = "{} created {} {} and {} work of art, {} didn't create {} {} work of art".format(person,
                                                                                                    preposition1,
                                                                                                    pred1,
                                                                                                    pred2,
                                                                                                    person,
                                                                                                    preposition3,
                                                                                                    pred3)
        all_sentences_1.append(sentence)

    all_sentences_1 = [sentence.split(",") + [1]
                       for sentence in all_sentences_1]

    # - 2) **c created a(n) P and Q work of art, c (d) didn't create a(n) W (P, Q)  work of art** (non-contradiction)

    all_sentences_2 = []
    for i in range(int(upper_bound)):
        person = get_new_item([], name_list)
        new_person = get_new_item([person], name_list)
        pred1 = get_new_item([], all_attributes)
        pred2 = get_new_item([pred1], all_attributes)
        pred3 = get_new_item([pred1, pred2], all_attributes)

        if pred1[0] in vowels:
            preposition1 = "an"
        else:
            preposition1 = "a"
        if pred2[0] in vowels:
            preposition2 = "an"
        else:
            preposition2 = "a"
        if pred3[0] in vowels:
            preposition3 = "an"
        else:
            preposition3 = "a"
        if i % 3 == 0:
            new_preposition = preposition1
            new_pred = pred1
            other = new_person
        elif i % 3 == 1:
            new_preposition = preposition2
            new_pred = pred2
            other = new_person
        else:
            new_preposition = preposition3
            new_pred = pred3
            other = person
        sentence = "{} created {} {} and {} work of art, {} didn't create {} {} work of art".format(person,
                                                                                                    preposition1,
                                                                                                    pred1,
                                                                                                    pred2,
                                                                                                    other,
                                                                                                    new_preposition,
                                                                                                    new_pred)
        all_sentences_2.append(sentence)

    all_sentences_2 = [sentence.split(",") + [0]
                       for sentence in all_sentences_2]

    np.random.shuffle(all_sentences_1)
    np.random.shuffle(all_sentences_2)

    size1 = len(all_sentences_1)
    size2 = len(all_sentences_2)

    all_sentences = all_sentences_1 + all_sentences_2

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

    df_train.to_csv("data/boolean5_train.csv", index=False)
    df_test.to_csv("data/boolean5_test.csv", index=False)


if __name__ == '__main__':
    boolean5()
