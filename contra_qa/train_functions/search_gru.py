import os
from random_search import naive_grid_search
from GRU import GRU

all_prefixes = ["GRU_boolean1",
                "GRU_boolean2",
                "GRU_boolean3",
                "GRU_boolean4",
                "GRU_boolean5",
                "GRU_boolean6",
                "GRU_boolean7",
                "GRU_boolean8",
                "GRU_boolean9",
                "GRU_boolean10",
                "GRU_boolean_AND",
                "GRU_boolean_OR",
                "GRU_boolean"]

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


# all_prefixes = ["GRU_boolean1_",
#                 "GRU_boolean2_"]

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

    best_acc, best_params, name = naive_grid_search(GRU,
                                                    10,
                                                    5,
                                                    train_data_path,
                                                    test_data_path,
                                                    prefix=prefix)
    with open("GRUresults_" + str(i + 1) + ".txt", "w") as file:
        file.write("results on {}\n".format(test))
        file.write("acc =  {:.3f}\n".format(best_acc))
        file.write("best_params =  {}\n".format(best_params))
        file.write("model path =  {}\n".format(name))
