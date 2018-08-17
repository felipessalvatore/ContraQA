import numpy as np
import pandas as pd
import os
from boolean1_neg import boolean1
from boolean2_S_and import boolean2
from boolean3_NP_and import boolean3
from boolean4_VP_and import boolean4
from boolean5_AP_and import boolean5
from boolean6_implicit_and import boolean6
from boolean7_S_or import boolean7
from boolean8_NP_or import boolean8
from boolean9_VP_or import boolean9
from boolean10_AP_or import boolean10

def create_all():
    boolean1()
    boolean2()
    boolean3()
    boolean4()
    boolean5()
    boolean6()
    boolean7()
    boolean8()
    boolean9()
    boolean10()

    # creating the AND dataset
    df2_tr = pd.read_csv("data/boolean2_train.csv")
    df3_tr = pd.read_csv("data/boolean3_train.csv")
    df4_tr = pd.read_csv("data/boolean4_train.csv")
    df5_tr = pd.read_csv("data/boolean5_train.csv")
    df6_tr = pd.read_csv("data/boolean6_train.csv")
    df2_te = pd.read_csv("data/boolean2_test.csv")
    df3_te = pd.read_csv("data/boolean3_test.csv")
    df4_te = pd.read_csv("data/boolean4_test.csv")
    df5_te = pd.read_csv("data/boolean5_test.csv")
    df6_te = pd.read_csv("data/boolean6_test.csv")

    train_and = [df2_tr, df3_tr, df4_tr, df5_tr, df6_tr]
    test_and = [df2_te, df3_te, df4_te, df5_te, df6_te]

    df_train_and = pd.concat(train_and)
    df_test_and = pd.concat(test_and)

    df_train_and = df_train_and.sample(frac=1).reset_index(drop=True)
    df_test_and = df_test_and.sample(frac=1).reset_index(drop=True)

    df_train_and = df_train_and.iloc[:10000]
    df_test_and = df_test_and.iloc[:1000]

    df_train_and.to_csv("data/boolean_AND_train.csv", index=False)
    df_test_and.to_csv("data/boolean_AND_test.csv", index=False)

    # creating the OR dataset
    df7_tr = pd.read_csv("data/boolean7_train.csv")
    df8_tr = pd.read_csv("data/boolean8_train.csv")
    df9_tr = pd.read_csv("data/boolean9_train.csv")
    df10_tr = pd.read_csv("data/boolean10_train.csv")
    df7_te = pd.read_csv("data/boolean7_test.csv")
    df8_te = pd.read_csv("data/boolean8_test.csv")
    df9_te = pd.read_csv("data/boolean9_test.csv")
    df10_te = pd.read_csv("data/boolean10_test.csv")

    train_or = [df7_tr, df8_tr, df9_tr, df10_tr]
    test_or = [df7_te, df8_te, df9_te, df10_te]

    df_train_or = pd.concat(train_or)
    df_test_or = pd.concat(test_or)

    df_train_or = df_train_or.sample(frac=1).reset_index(drop=True)
    df_test_or = df_test_or.sample(frac=1).reset_index(drop=True)

    df_train_or = df_train_or.iloc[:10000]
    df_test_or = df_test_or.iloc[:1000]

    df_train_or.to_csv("data/boolean_OR_train.csv", index=False)
    df_test_or.to_csv("data/boolean_OR_test.csv", index=False)

    # creating the boolean dataset
    boolean_train = [df_train_and, df_train_or]
    boolean_test = [df_test_and, df_test_or]

    df_boolean_train = pd.concat(boolean_train)
    df_boolean_test = pd.concat(boolean_test)

    df_boolean_train = df_boolean_train.sample(frac=1).reset_index(drop=True)
    df_boolean_test = df_boolean_test.sample(frac=1).reset_index(drop=True)

    df_boolean_train = df_boolean_train.iloc[:10000]
    df_boolean_test = df_boolean_test.iloc[:1000]

    df_boolean_train.to_csv("data/boolean_train.csv", index=False)
    df_boolean_test.to_csv("data/boolean_test.csv", index=False)


if __name__ == '__main__':
    create_all()
