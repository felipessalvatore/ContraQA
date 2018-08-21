import os
from random_search import naive_grid_search
from RNN import RNN

all_prefixes = ["RNN_boolean1",
                "RNN_boolean2",
                "RNN_boolean3",
                "RNN_boolean4",
                "RNN_boolean5",
                "RNN_boolean6",
                "RNN_boolean7",
                "RNN_boolean8",
                "RNN_boolean9",
                "RNN_boolean10",
                "RNN_boolean_AND",
                "RNN_boolean_OR",
                "RNN_boolean"]

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
                  "boolean_train.csv"]

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
                 "boolean_test.csv"]


# all_prefixes = ["RNN_boolean1_",
#                 "RNN_boolean2_"]

# all_train_data = ["boolean1_train.csv",
#                   "boolean2_train.csv"]

# all_test_data = ["boolean1_test.csv",
#                  "boolean2_test.csv"]

for i, (prefix, train, test) in enumerate(zip(all_prefixes,
                                              all_train_data,
                                              all_test_data)):
    print(prefix, "\n")
    train_data_path = os.path.join("data", train)
    test_data_path = os.path.join("data", test)
    print(train_data_path)
    print(test_data_path)

    best_acc, best_params, name = naive_grid_search(RNN,
                                                    10,
                                                    5,
                                                    train_data_path,
                                                    test_data_path,
                                                    prefix=prefix)
    with open("RNNresults_" + str(i + 1) + ".txt", "w") as file:
        file.write("results on {}\n".format(test))
        file.write("acc =  {:.3f}\n".format(best_acc))
        file.write("best_params =  {}\n".format(best_params))
        file.write("model path =  {}\n".format(name))
