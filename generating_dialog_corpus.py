import pandas as pd
import os
from contra_qa.text_processing.functions import simple_pre_process_text_df
from contra_qa.text_generation.boolean_data_gen import create_all


def csv_to_FBdialog(df,
                    out_path,
                    select_label="label",
                    question="There is a contradiction here?"):
    """
    csv to Facebook Dialog format
    """
    sentence1 = list(df["sentence1"])
    sentence2 = list(df["sentence2"])
    labels = list(df[select_label])
    count = 1
    with open(out_path, "w") as file:
        for triple in zip(sentence1, sentence2, labels):
            file.write(str(count) + " " + triple[0] + "\n")
            count += 1
            file.write(str(count) + " " + triple[1] + "\n")
            count += 1
            file.write(str(count) + " " + question + "\t{}\t1\n".format(triple[2].lower())) # noqa 
            count += 1
            if count == 16:
                count = 1


all_train_data = ["boolean1_train.csv",
                  "boolean2_train.csv",
                  "boolean3_train.csv",
                  "boolean4_train.csv",
                  "boolean5_train.csv",
                  "boolean6_train.csv",
                  "boolean7_train.csv",
                  "boolean8_train.csv",
                  "boolean9_train.csv",
                  "boolean10_train.csv",
                  "boolean_AND_train.csv",
                  "boolean_OR_train.csv",
                  "boolean_train.csv",
                  "boolean3_inf_train.csv",
                  "boolean4_inf_train.csv",
                  "boolean5_inf_train.csv",
                  "boolean8_inf_train.csv",
                  "boolean9_inf_train.csv",
                  "boolean10_inf_train.csv",
                  "boolean_AND_inf_train.csv",
                  "boolean_OR_inf_train.csv",
                  "boolean_inf_train.csv"]


all_test_data = ["boolean1_test.csv",
                 "boolean2_test.csv",
                 "boolean3_test.csv",
                 "boolean4_test.csv",
                 "boolean5_test.csv",
                 "boolean6_test.csv",
                 "boolean7_test.csv",
                 "boolean8_test.csv",
                 "boolean9_test.csv",
                 "boolean10_test.csv",
                 "boolean_AND_test.csv",
                 "boolean_OR_test.csv",
                 "boolean_test.csv",
                 "boolean3_inf_test.csv",
                 "boolean4_inf_test.csv",
                 "boolean5_inf_test.csv",
                 "boolean8_inf_test.csv",
                 "boolean9_inf_test.csv",
                 "boolean10_inf_test.csv",
                 "boolean_AND_inf_test.csv",
                 "boolean_OR_inf_test.csv",
                 "boolean_inf_test.csv"]


all_valid_path = ["boolean1_valid",
                  "boolean2_valid",
                  "boolean3_valid",
                  "boolean4_valid",
                  "boolean5_valid",
                  "boolean6_valid",
                  "boolean7_valid",
                  "boolean8_valid",
                  "boolean9_valid",
                  "boolean10_valid",
                  "boolean_AND_valid",
                  "boolean_OR_valid",
                  "boolean_valid",
                  "boolean3_inf_valid",
                  "boolean4_inf_valid",
                  "boolean5_inf_valid",
                  "boolean8_inf_valid",
                  "boolean9_inf_valid",
                  "boolean10_inf_valid",
                  "boolean_AND_inf_valid",
                  "boolean_OR_inf_valid",
                  "boolean_inf_valid"]


def main():
    if not os.path.exists("dialog_data"):
        os.makedirs("dialog_data")

    if not os.path.exists("fixed_data"):
        print("Generating data \n")
        create_all()

    for tra, te, val in zip(all_train_data, all_test_data, all_valid_path):

        train_data_path = os.path.join("fixed_data", tra)
        test_data_path = os.path.join("fixed_data", te)
        out_path_train = os.path.join("dialog_data", tra)[:-3] + "txt"
        out_path_test = os.path.join("dialog_data", te)[:-3] + "txt"
        out_path_valid = os.path.join("dialog_data", val) + ".txt"

        df_train = pd.read_csv(train_data_path)
        df_train = df_train.sample(frac=1).reset_index(drop=True)

        dftrain = df_train.iloc[:8000]
        dfvalid = df_train.iloc[8000:]

        dftest = pd.read_csv(test_data_path)

        simple_pre_process_text_df(dftrain, "sentence1")
        simple_pre_process_text_df(dftrain, "sentence2")
        dftrain["label"] = dftrain["label"].apply(lambda x: "yes" if x==1 else "no") # noqa
        simple_pre_process_text_df(dfvalid, "sentence1")
        simple_pre_process_text_df(dfvalid, "sentence2")
        dfvalid["label"] = dfvalid["label"].apply(lambda x: "yes" if x==1 else "no") # noqa
        simple_pre_process_text_df(dftest, "sentence1")
        simple_pre_process_text_df(dftest, "sentence2")
        dftest["label"] = dftest["label"].apply(lambda x: "yes" if x==1 else "no") # noqa
        csv_to_FBdialog(dftrain, out_path_train)
        csv_to_FBdialog(dfvalid, out_path_valid)
        csv_to_FBdialog(dftest, out_path_test)


if __name__ == '__main__':
    main()
