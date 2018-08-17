from util import training_loop_text_classification, get_data
from RNN import RNN
from DataHolder import DataHolder
from RNNConfig import RNNConfig
import os
import inspect
import sys
import torch
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

    valid_bach = next(iter(current_data.valid_iter))
    acc, _, _ = model.evaluate_bach(valid_bach)

    return acc


def eval_RNN_on_test(train_data_path,
                     test_data_path,
                     pkl_path,
                     epochs,
                     embedding_dim,
                     learning_rate,
                     momentum):
    """
    Evaluating the RNN on the test data
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
    model.load_state_dict(torch.load(pkl_path))
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
                  epoch_bounds=[1, 10],
                  embedding_dim_bounds=[10, 500],
                  learning_rate_bounds=[0, 1],
                  momentum_bounds=[0, 1],
                  verbose=True,
                  prefix=""):
    """
    function to perform random search
    """
    all_acc = []
    all_hyper_params = []
    for i in range(trials):
        if verbose:
            print("=== random search: ({}/{})\n".format(i + 1, trials))
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

        if not os.path.exists("./tmp_pkl"):
            os.makedirs("tmp_pkl/")

        name = "epochs_{}_embedding_dim_{}_learning_rate_{:.3f}_momentum_{:.3f}.pkl".format(hyper_dict["epochs"], # noqa
                                                                                            hyper_dict["embedding_dim"], # noqa
                                                                                            hyper_dict["learning_rate"], # noqa
                                                                                            hyper_dict["momentum"]) # noqa
        name = os.path.join("tmp_pkl", prefix + name)

        acc = run_train_RNN_on_params(train_data_path,
                                      test_data_path,
                                      name,
                                      epochs,
                                      embedding_dim,
                                      learning_rate,
                                      momentum,
                                      verbose=False)
        if verbose:
            print("====== dict", hyper_dict)
            print("====== acc", acc)
        all_acc.append(acc)
        all_hyper_params.append(hyper_dict)

    return all_acc, all_hyper_params, name


def naive_grid_search(search_trials,
                      random_trials,
                      train_data_path,
                      test_data_path,
                      epoch_bounds=[1, 10],
                      embedding_dim_bounds=[10, 500],
                      learning_rate_bounds=[0, 1],
                      momentum_bounds=[0, 1],
                      verbose=True,
                      prefix=""):
    """
    function to perform a naive version of grid search

    returns the test acc of the best model (best in the sense
    of higher valid acc)
    """
    epoch_bounds = epoch_bounds
    embedding_dim_bounds = embedding_dim_bounds
    learning_rate_bounds = learning_rate_bounds
    momentum_bounds = momentum_bounds
    best_acc = 0
    best_params = None
    model_path = None

    for i in range(search_trials):
        print("grid_search ({}/{})\n".format(i + 1, search_trials))

        all_acc, all_hyper_params, name = random_search(random_trials,
                                                        train_data_path,
                                                        test_data_path,
                                                        epoch_bounds,
                                                        embedding_dim_bounds,
                                                        learning_rate_bounds,
                                                        momentum_bounds,
                                                        verbose=verbose,
                                                        prefix=prefix)
        best_i = np.argmax(all_acc)
        current_acc = all_acc[best_i]
        current_dict = all_hyper_params[best_i]
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
                model_path = name

    test_acc = eval_RNN_on_test(train_data_path,
                                test_data_path,
                                model_path,
                                epochs=current_dict["epochs"],
                                embedding_dim=current_dict["embedding_dim"], # noqa
                                learning_rate=current_dict["learning_rate"], # noqa
                                momentum=current_dict["momentum"])

    return test_acc, best_params, model_path


# train_data_path = os.path.join(parentdir,
#                                "text_generation",
#                                "data",
#                                "boolean1_train.csv")

# test_data_path = os.path.join(parentdir,
#                               "text_generation",
#                               "data",
#                               "boolean1_test.csv")


# best_acc, best_params, name = naive_grid_search(5,
#                                                 2,
#                                                 train_data_path,
#                                                 test_data_path,
#                                                 epoch_bounds=[2, 4])

# print(best_acc, best_params, name)
# assert best_acc > 0.5
