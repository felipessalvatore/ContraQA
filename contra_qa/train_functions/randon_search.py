from util import training_loop_text_classification, get_data
from RNN import RNN
from DataHolder import DataHolder
from RNNConfig import RNNConfig
import os
import inspect
import sys
import numpy as np

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def run_train_RNN_on_params(train_data_path,
                            test_data_path,
                            pkl_path,
                            epochs,
                            embedding_dim,
                            learning_rate,
                            momentum,
                            verbose):
    """
    Using the params
            epochs,
            embedding_dim,
            learning_rate,
            momentum
    this functions trains a model and outputs the test accuracy
    """

    data_folder = "."
    TEXT, LABEL, train, valid, test = get_data(train_data_path,
                                               test_data_path,
                                               data_folder)

    current_config = RNNConfig(vocab_size=len(TEXT.vocab),
                               output_dim=len(LABEL.vocab),
                               epochs=epochs,
                               embedding_dim=embedding_dim,
                               learning_rate=learning_rate,
                               momentum=momentum)

    current_data = DataHolder(current_config,
                              train,
                              valid,
                              test)

    model = RNN(current_config)

    training_loop_text_classification(model,
                                      current_config,
                                      current_data,
                                      pkl_path,
                                      verbose=verbose)

    test_bach = next(iter(current_data.test_iter))
    acc, _, _ = model.evaluate_bach(test_bach)

    return acc


def get_random_discrete_param(lower_bound, upper_bound):

    return np.random.randint(lower_bound, upper_bound)


def get_random_cont_param(lower_bound=0, upper_bound=1):

    return np.random.uniform(lower_bound, upper_bound)


def random_search(trials,
                  train_data_path,
                  test_data_path,
                  pkl_path,
                  epoch_bounds=[1, 10],
                  embedding_dim_bounds=[10, 500],
                  learning_rate_bounds=[0, 1],
                  momentum_bounds=[0, 1],
                  verbose=True):
    """
    function to perform random search
    """
    all_acc = []
    all_hyper_params = []
    for i in range(trials):
        if verbose:
            print("random search: ({}/{})\n".format(i + 1, trials))
        epochs = get_random_discrete_param(epoch_bounds[0], epoch_bounds[1])
        embedding_dim = get_random_discrete_param(embedding_dim_bounds[0],
                                                  embedding_dim_bounds[1])
        learning_rate = get_random_cont_param(learning_rate_bounds[0],
                                              learning_rate_bounds[1])
        momentum = get_random_cont_param(momentum_bounds[0],
                                         momentum_bounds[1])
        hyper_dict = {"epochs": epochs,
                      "embedding_dim": embedding_dim,
                      "learning_rate": learning_rate,
                      "momentum": momentum}
        acc = run_train_RNN_on_params(train_data_path,
                                      test_data_path,
                                      pkl_path,
                                      epochs,
                                      embedding_dim,
                                      learning_rate,
                                      momentum,
                                      verbose=False)
        if verbose:
            print("dict", hyper_dict)
            print("acc", acc)
        all_acc.append(acc)
        all_hyper_params.append(hyper_dict)

    return all_acc, all_hyper_params


def naive_grid_search(search_trials,
                      random_trials,
                      train_data_path,
                      test_data_path,
                      pkl_path,
                      epoch_bounds=[1, 10],
                      embedding_dim_bounds=[10, 500],
                      learning_rate_bounds=[0, 1],
                      momentum_bounds=[0, 1],
                      verbose=True):
    """
    function to perform random search
    """
    epoch_bounds = epoch_bounds
    embedding_dim_bounds = embedding_dim_bounds
    learning_rate_bounds = learning_rate_bounds
    momentum_bounds = momentum_bounds
    best_acc = 0
    best_params = None

    for i in range(search_trials):
        print("grid_search ({}/{})\n".format(i + 1, search_trials))

        all_acc, all_hyper_params = random_search(random_trials,
                                                  train_data_path,
                                                  test_data_path,
                                                  pkl_path,
                                                  epoch_bounds,
                                                  embedding_dim_bounds,
                                                  learning_rate_bounds,
                                                  momentum_bounds,
                                                  verbose=verbose)
        tuples = list(zip(all_acc, all_hyper_params))
        tuples.sort(reverse=True)
        current_acc, current_dict = tuples[0]
        if best_acc < current_acc:
                epoch_bounds = [epoch_bounds[0],
                                current_dict["epochs"] + 1]
                embedding_dim_bounds = [embedding_dim_bounds[0],
                                        current_dict["embedding_dim"] + 1]
                learning_rate_bounds = [learning_rate_bounds[0],
                                        current_dict["learning_rate"]]
                momentum_bounds = [momentum_bounds[0],
                                   current_dict["momentum"]]
                best_acc = current_acc
                best_params = current_dict
    return best_acc, best_params


# train_data_path = os.path.join(parentdir,
#                                "text_generation",
#                                "data",
#                                "boolean1_train.csv")

# test_data_path = os.path.join(parentdir,
#                               "text_generation",
#                               "data",
#                               "boolean1_test.csv")


# best_acc, best_params = naive_grid_search(5,
#                                           2,
#                                           train_data_path,
#                                           test_data_path,
#                                           "rnn_boolean1.pkl",
#                                           epoch_bounds=[2, 4])

# print(best_acc, best_params)
# assert best_acc > 0.9
