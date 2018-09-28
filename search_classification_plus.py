import os
import argparse
from contra_qa.text_generation.boolean_data_gen import create_all
from contra_qa.train_functions.RNN import RNN
from contra_qa.train_functions.LSTM import LSTM
from contra_qa.train_functions.GRU import GRU
from contra_qa.train_functions.random_search import naive_grid_search


all_prefixes = ["boolean3_plus",
                "boolean4_plus",
                "boolean5_plus"]

all_train_data = ["boolean3_plus_train.csv",
                  "boolean4_plus_train.csv",
                  "boolean5_plus_train.csv"]

all_test_data = ["boolean3_plus_test.csv",
                 "boolean4_plus_test.csv",
                 "boolean5_plus_test.csv"]


def search(all_prefixes,
           all_train_data,
           all_test_data,
           Model,
           model_name,
           search_trails,
           random_trails,
           acc_bound):
    if not os.path.exists("data"):
        print("Generating data \n")
        # create_all()
    if not os.path.exists("results"):
        os.makedirs("results")

    all_prefixes = [model_name + prefix for prefix in all_prefixes]
    best_pkls = []

    for i, (prefix, train, test) in enumerate(zip(all_prefixes,
                                              all_train_data,
                                              all_test_data)):
            print(prefix, "\n")
            train_data_path = os.path.join("data", train)
            test_data_path = os.path.join("data", test)
            print(train_data_path)
            print(test_data_path)

            best_acc, best_params, name = naive_grid_search(Model,
                                                            search_trails,
                                                            random_trails,
                                                            train_data_path,
                                                            test_data_path,
                                                            prefix=prefix,
                                                            acc_bound=acc_bound) # noqa
            path = os.path.join("results", prefix + "_results.txt") # noqa
            best_pkls.append(name)
            with open(path, "w") as file:
                file.write("results on {}\n".format(test))
                file.write("acc =  {:.3f}\n".format(best_acc))
                file.write("best_params =  {}\n".format(best_params))
                file.write("model path =  {}\n".format(name))

    if os.path.exists("tmp_pkl"):
        for file in os.listdir("tmp_pkl"):
            new_file = os.path.join("tmp_pkl", file)
            if new_file not in best_pkls:
                os.remove(new_file)


def main():
    msg = """Function to perform a grid search using one hidden layer recurrent model.\n

            Models = RNN, GRU, LSTM\n

            Tasks = 
                    3) boolean3: NP conjoined by and PLUS\n
                    4) boolean4: VP conjoined by and PLUS\n
                    5) boolean5: AP conjoined by and PLUS\n"""
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-m",
                    "--model",
                    type=str,
                    default="RNN",
                    help="model type: 'RNN', 'GRU', 'LSTM' (default=RNN)") # noqa
    parser.add_argument("-st",
                    "--search_trails",
                    type=int,
                    default=10,
                    help="number of times to call the grid seach funtion(default=10)") # noqa
    parser.add_argument("-rt",
                    "--random_trails",
                    type=int,
                    default=5,
                    help="number of times to call the random seach funtion(default=5)") # noqa
    parser.add_argument("-ab",
                    "--acc_bound",
                    type=float,
                    default=0.9,
                    help=" upper bound for the accuracy of each task (default=0.9)") # noqa
    parser.add_argument("-s",
                    "--start",
                    type=int,
                    default=1,
                    help="position of the first task to be searched -- min=1 (default=1)") # noqa
    parser.add_argument("-e",
                    "--end",
                    type=int,
                    default=13,
                    help="position of the last task to be searched -- max=3 (default=13)") # noqa

    args = parser.parse_args()
    models_and_names = {"RNN": RNN, "GRU": GRU, "LSTM": LSTM}
    msg = "not a valid mode"
    user_model = args.model.upper()
    assert user_model in models_and_names
    Model = models_and_names[user_model]
    model_name = user_model + "_"
    search_trails = args.search_trails
    random_trails = args.random_trails
    acc_bound = args.acc_bound
    start = args.start - 1
    end = args.end
    all_prefixes_cut = all_prefixes[start: end]
    all_train_data_cut = all_train_data[start: end]
    all_test_data_cut = all_test_data[start: end]
    search(all_prefixes_cut,
           all_train_data_cut,
           all_test_data_cut,
           Model,
           model_name,
           search_trails,
           random_trails,
           acc_bound)


if __name__ == '__main__':
    main()
