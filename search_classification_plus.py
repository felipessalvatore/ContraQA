import os
import argparse
import time
from contra_qa.text_generation.boolean_data_gen import create_all
from contra_qa.train_functions.RNN import RNN
from contra_qa.train_functions.LSTM import LSTM
from contra_qa.train_functions.GRU import GRU
from contra_qa.train_functions.random_search import naive_grid_search
from contra_qa.train_functions.util import timeSince


# all_prefixes = ["boolean3_plus_",
#                 "boolean4_plus_",
#                 "boolean5_plus_",
#                 "boolean3_control_",
#                 "boolean4_control_",
#                 "boolean5_control_",
#                 "boolean8_control_",
#                 "boolean9_control_",
#                 "boolean10_control_"]

# all_train_data = ["boolean3_plus_train.csv",
#                   "boolean4_plus_train.csv",
#                   "boolean5_plus_train.csv",
#                   "boolean3_control_train.csv",
#                   "boolean4_control_train.csv",
#                   "boolean5_control_train.csv",
#                   "boolean8_control_train.csv",
#                   "boolean9_control_train.csv",
#                   "boolean10_control_train.csv"]

# all_test_data = ["boolean3_plus_test.csv",
#                  "boolean4_plus_test.csv",
#                  "boolean5_plus_test.csv",
#                  "boolean3_control_test.csv",
#                  "boolean4_control_test.csv",
#                  "boolean5_control_test.csv",
#                  "boolean8_control_test.csv",
#                  "boolean9_control_test.csv",
#                  "boolean10_control_test.csv"]

# all_prefixes = ["boolean_control_"]
# all_train_data = ["boolean_control_train.csv"]
# all_test_data = ["boolean_control_test.csv"]

# all_prefixes = ["boolean_AND_control_"]
# all_train_data = ["boolean_AND_control_train.csv"]
# all_test_data = ["boolean_AND_control_test.csv"]


# all_prefixes = ["boolean_OR_control_"]
# all_train_data = ["boolean_OR_control_train.csv"]
# all_test_data = ["boolean_OR_control_test.csv"]

# all_prefixes = ["boolean1_"]
# all_train_data = ["boolean1_train.csv"]
# all_test_data = ["boolean1_test.csv"]


all_prefixes = ["boolean3_control_",
                "boolean4_control_",
                "boolean5_control_",
                "boolean8_control_",
                "boolean9_control_",
                "boolean10_control_",
                "boolean_AND_control_",
                "boolean_OR_control_",
                "boolean_control_"]

all_train_data = ["boolean3_control_train.csv",
                  "boolean4_control_train.csv",
                  "boolean5_control_train.csv",
                  "boolean8_control_train.csv",
                  "boolean9_control_train.csv",
                  "boolean10_control_train.csv",
                  "boolean_AND_control_train.csv",
                  "boolean_OR_control_train.csv",
                  "boolean_control_train.csv"]

all_test_data = ["boolean3_control_test.csv",
                 "boolean4_control_test.csv",
                 "boolean5_control_test.csv",
                 "boolean8_control_test.csv",
                 "boolean9_control_test.csv",
                 "boolean10_control_test.csv",
                 "boolean_AND_control_test.csv",
                 "boolean_OR_control_test.csv",
                 "boolean_control_test.csv"]

# all_prefixes = ["boolean4_control_"]
# all_train_data = ["boolean4_control_train.csv"]
# all_test_data = ["boolean4_control_test.csv"]


def search(all_prefixes,
           all_train_data,
           all_test_data,
           Model,
           model_name,
           search_trails,
           random_trails,
           acc_bound,
           load_emb,
           bidirectional,
           freeze_emb,
           opt):
    if not os.path.exists("data"):
        print("Generating data \n")
        create_all()
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
                                                        acc_bound=acc_bound,  # noqa
                                                        load_emb=load_emb,
                                                        bidirectional=bidirectional,  # noqa
                                                        freeze_emb=freeze_emb,
                                                        opt=opt)  # noqa
        path = os.path.join("results", prefix + "_results.txt")  # noqa
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
                    5) boolean5: AP conjoined by and PLUS\n
                    8) boolean3: NP conjoined by or PLUS\n
                    9) boolean4: VP conjoined by or PLUS\n
                    10) boolean5: AP conjoined by or PLUS\n

            List of pre trained word embeddings

                     None : None
                     charngram :  charngram.100d
                     fasttextEn :  fasttext.en.300d
                     fasttextSimple :  fasttext.simple.300d
                     glove42 :  glove.42B.300d
                     glove84 :  glove.840B.300d
                     gloveTwitter25 :  glove.twitter.27B.25d
                     gloveTwitter50 :  glove.twitter.27B.50d
                     gloveTwitter100 :  glove.twitter.27B.100d
                     gloveTwitter200 :  glove.twitter.27B.200d
                     glove6b_80 :  glove.6B.50d
                     glove6b_100 :  glove.6B.100d
                     glove6b_200 :  glove.6B.200d
                     glove6b_300 :  glove.6B.300d
                    """
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("-m",
                    "--model",
                    type=str,
                    default="RNN",
                    help="model type: 'RNN', 'GRU', 'LSTM' (default=RNN)")  # noqa
    parser.add_argument("-st",
                    "--search_trails",
                    type=int,
                    default=10,
                    help="number of times to call the grid seach funtion(default=10)")  # noqa
    parser.add_argument("-rt",
                    "--random_trails",
                    type=int,
                    default=5,
                    help="number of times to call the random seach funtion(default=5)")  # noqa
    parser.add_argument("-ab",
                    "--acc_bound",
                    type=float,
                    default=1.0,
                    help=" upper bound for the accuracy of each task (default=1.0)")  # noqa
    parser.add_argument("-s",
                    "--start",
                    type=int,
                    default=1,
                    help="position of the first task to be searched -- min=1 (default=1)")  # noqa
    parser.add_argument("-e",
                    "--end",
                    type=int,
                    default=13,
                    help="position of the last task to be searched -- max=3 (default=13)")  # noqa
    parser.add_argument("-em",
                    "--embedding",
                    type=str,
                    default="None",
                    help="pre trained word embedding (default=None)")  # noqa
    parser.add_argument("-bi",
                        "--bidirectional",
                        action="store_true",
                        default=False,
                        help="Use bidirectional rnn (default=False)")
    parser.add_argument("-f",
                        "--freeze_emb",
                        action="store_true",
                        default=False,
                        help="freeze embedding layer (default=False)")
    parser.add_argument("-o",
                    "--optmizer",
                    type=str,
                    default="sgd",
                    help="torch optmizer: 'sgd', 'adam', 'adagrad' 'rmsprop' (default=sgd)")  # noqa
    args = parser.parse_args()
    models_and_names = {"RNN": RNN, "GRU": GRU, "LSTM": LSTM}
    embedding_and_names = {"None": None,
                           "charngram": "charngram.100d",
                           "fasttextEn": "fasttext.en.300d",
                           "fasttextSimple": "fasttext.simple.300d",
                           "glove42": "glove.42B.300d",
                           "glove84": "glove.840B.300d",
                           "gloveTwitter25": "glove.twitter.27B.25d",
                           "gloveTwitter50": "glove.twitter.27B.50d",
                           "gloveTwitter100": "glove.twitter.27B.100d",
                           "gloveTwitter200": "glove.twitter.27B.200d",
                           "glove6b_80": "glove.6B.50d",
                           "glove6b_100": "glove.6B.100d",
                           "glove6b_200": "glove.6B.200d",
                           "glove6b_300": "glove.6B.300d"}
    msg = "not a valid model"
    user_model = args.model.upper()
    assert user_model in models_and_names, msg
    opt = args.optmizer.lower().strip()
    msg = "not a valid opt"
    all_opts = ['sgd', 'adam', 'adagrad', 'rmsprop']
    assert opt in all_opts, msg

    Model = models_and_names[user_model]
    load_emb = embedding_and_names[args.embedding]
    model_name = user_model + "_"
    search_trails = args.search_trails
    random_trails = args.random_trails
    acc_bound = args.acc_bound
    start = args.start - 1
    end = args.end
    bidirectional = args.bidirectional
    all_prefixes_cut = all_prefixes[start: end]
    all_train_data_cut = all_train_data[start: end]
    all_test_data_cut = all_test_data[start: end]
    freeze = args.freeze_emb
    opt = args.optmizer

    start = time.time()

    search(all_prefixes_cut,
           all_train_data_cut,
           all_test_data_cut,
           Model,
           model_name,
           search_trails,
           random_trails,
           acc_bound,
           load_emb,
           bidirectional,
           freeze,
           opt)

    n_exp = len(all_prefixes_cut) * search_trails * random_trails

    print("time after <= {} experiments: {}".format(n_exp, timeSince(start)))


if __name__ == '__main__':
    main()
