import os
import inspect
import sys
from randon_search import naive_grid_search


almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

all_prefixes = ["rnn_boolean1",
                "rnn_boolean2",
                "rnn_boolean3",
                "rnn_boolean4",
                "rnn_boolean5",
                "rnn_boolean6",
                "rnn_boolean7",
                "rnn_boolean8",
                "rnn_boolean9",
                "rnn_boolean10",
                "rnn_boolean_AND",
                "rnn_boolean_OR",
                "rnn_boolean"]

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


# all_prefixes = ["rnn_boolean1_",
#                 "rnn_boolean2_"]

# all_train_data = ["boolean1_train.csv",
#                   "boolean2_train.csv"]

# all_test_data = ["boolean1_test.csv",
#                  "boolean2_test.csv"]

data_path = os.path.join(parentdir,
                         "text_generation",
                         "data")

for i, (prefix, train, test) in enumerate(zip(all_prefixes,
                                              all_train_data,
                                              all_test_data)):
    print(prefix, "\n")
    train_data_path = os.path.join(data_path, train)
    test_data_path = os.path.join(data_path, test)
    print(train_data_path)
    print(test_data_path)

    best_acc, best_params, name = naive_grid_search(10,
                                                    4,
                                                    train_data_path,
                                                    test_data_path,
                                                    prefix=prefix)
    print(best_acc, best_params, name)
    with open("results_" + str(i + 1) + ".txt", "w") as file:
        file.write("results on {}\n".format(test))
        file.write("acc =  {:.3f}\n".format(best_acc))
        file.write("best_params =  {}\n".format(best_params))
        file.write("model path =  {}\n".format(name))
