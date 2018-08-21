import os
import torch
import numpy as np

try:
    from util import training_loop_text_classification, get_data # noqa
except ImportError:
    from contra_qa.train_functions.util import training_loop_text_classification, get_data # noqa


try:
    from RNNConfig import RNNConfig # noqa
except ImportError:
    from contra_qa.train_functions.RNNConfig import RNNConfig # noqa


try:
    from DataHolder import DataHolder # noqa
except ImportError:
    from contra_qa.train_functions.DataHolder import DataHolder # noqa


def train_model_on_params(Model,
                          train_data_path,
                          test_data_path,
                          pkl_path,
                          epochs,
                          embedding_dim,
                          rnn_dim,
                          learning_rate,
                          momentum):
    """
    Train model on param


    :param model: recurrent model
    :type model: RNN, LSTM, GRU
    :param train_data_path: path to train
    :type train_data_path: str
    :param test_data_path: path to test
    :type test_data_path: str
    :param pkl_path: path to model
    :type pkl_path: str
    :param epochs: number of epochs
    :type epochs: int
    :param embedding_dim: embedding dimention
    :type embedding_dim: int
    :param rnn_dim: rnn hidden size dimention
    :type rnn_dim: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param momentum: momentum param
    :type momentum: float


    :return: accuracy on the valid data
    :rtype: float
    """
    TEXT, LABEL, train, valid, test = get_data(train_data_path,
                                               test_data_path)

    current_config = RNNConfig(vocab_size=len(TEXT.vocab),
                               output_dim=len(LABEL.vocab),
                               epochs=epochs,
                               rnn_dim=rnn_dim,
                               embedding_dim=embedding_dim,
                               learning_rate=learning_rate,
                               momentum=momentum)
    model = Model(current_config)

    current_data = DataHolder(current_config,
                              train,
                              valid,
                              test)

    training_loop_text_classification(model,
                                      current_config,
                                      current_data,
                                      pkl_path,
                                      verbose=False)

    valid_bach = next(iter(current_data.valid_iter))
    acc, _, _ = model.evaluate_bach(valid_bach)

    return acc


def eval_model_on_test(Model,
                       train_data_path,
                       test_data_path,
                       pkl_path,
                       epochs,
                       embedding_dim,
                       rnn_dim,
                       learning_rate,
                       momentum):
    """
    Eval model on param

    :param model: recurrent model
    :type model: RNN, LSTM, GRU
    :param train_data_path: path to train
    :type train_data_path: str
    :param test_data_path: path to test
    :type test_data_path: str
    :param pkl_path: path to model
    :type pkl_path: str
    :param epochs: number of epochs
    :type epochs: int
    :param embedding_dim: embedding dimention
    :type embedding_dim: int
    :param rnn_dim: rnn hidden size dimention
    :type rnn_dim: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param momentum: momentum param
    :type momentum: float


    :return: accuracy on the valid data
    :rtype: float
    """

    TEXT, LABEL, train, valid, test = get_data(train_data_path,
                                               test_data_path)

    current_config = RNNConfig(vocab_size=len(TEXT.vocab),
                               output_dim=len(LABEL.vocab),
                               epochs=epochs,
                               rnn_dim=rnn_dim,
                               embedding_dim=embedding_dim,
                               learning_rate=learning_rate,
                               momentum=momentum)

    current_data = DataHolder(current_config,
                              train,
                              valid,
                              test)

    model = Model(current_config)
    model.load_state_dict(torch.load(pkl_path))
    test_bach = next(iter(current_data.test_iter))
    acc, _, _ = model.evaluate_bach(test_bach)

    return acc


def get_random_discrete_param(lower_bound, upper_bound):

    return np.random.randint(lower_bound, upper_bound)


def get_random_cont_param(lower_bound=0, upper_bound=1):

    return np.random.uniform(lower_bound, upper_bound)


def random_search(Model,
                  trials,
                  train_data_path,
                  test_data_path,
                  epoch_bounds=[1, 10],
                  embedding_dim_bounds=[10, 500],
                  rnn_dim_bounds=[10, 500],
                  learning_rate_bounds=[0, 1],
                  momentum_bounds=[0, 1],
                  verbose=True,
                  prefix=""):
    """
    Train model in n trails on random params


    :param model: recurrent model
    :type model: RNN, LSTM, GRU
    :param train_data_path: path to train
    :type train_data_path: str
    :param test_data_path: path to test
    :type test_data_path: str
    :param pkl_path: path to model
    :type pkl_path: str
    :param epochs: number of epochs
    :type epochs: int
    :param embedding_dim: embedding dimention
    :type embedding_dim: int
    :param rnn_dim: rnn hidden size dimention
    :type rnn_dim: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param momentum: momentum param
    :type momentum: float


    :return: list  of accuracy, list of hyperperams, list of pkl paths
    :rtype: [float], [dict], [str]
    """
    all_acc = []
    all_hyper_params = []
    all_names = []
    for i in range(trials):
        if verbose:
            print("=== random search: ({}/{})\n".format(i + 1, trials))
        epochs = get_random_discrete_param(epoch_bounds[0], epoch_bounds[1])
        embedding_dim = get_random_discrete_param(embedding_dim_bounds[0],
                                                  embedding_dim_bounds[1])
        rnn_dim = get_random_discrete_param(rnn_dim_bounds[0],
                                            rnn_dim_bounds[1])
        learning_rate = get_random_cont_param(learning_rate_bounds[0],
                                              learning_rate_bounds[1])
        momentum = get_random_cont_param(momentum_bounds[0],
                                         momentum_bounds[1])
        hyper_dict = {"epochs": epochs,
                      "embedding_dim": embedding_dim,
                      "rnn_dim": rnn_dim,
                      "learning_rate": learning_rate,
                      "momentum": momentum}

        if not os.path.exists("tmp_pkl"):
            os.makedirs("tmp_pkl/")

        name = "epochs_{}_embedding_dim_{}_rnn_dim_{}_learning_rate_{:.3f}_momentum_{:.3f}".format(hyper_dict["epochs"], # noqa
                                                                                        hyper_dict["embedding_dim"], # noqa
                                                                                        hyper_dict["rnn_dim"], # noqa
                                                                                        hyper_dict["learning_rate"], # noqa
                                                                                        hyper_dict["momentum"]) # noqa
        name = name.replace(".", "p") + ".pkl"
        name = os.path.join("tmp_pkl", prefix + name)

        acc = train_model_on_params(Model,
                                    train_data_path,
                                    test_data_path,
                                    name,
                                    epochs,
                                    embedding_dim,
                                    rnn_dim,
                                    learning_rate,
                                    momentum)
        if verbose:
            print("====== dict", hyper_dict)
            print("====== acc", acc)
        all_acc.append(acc)
        all_hyper_params.append(hyper_dict)
        all_names.append(name)

    return all_acc, all_hyper_params, all_names


def naive_grid_search(Model,
                      search_trials,
                      random_trials,
                      train_data_path,
                      test_data_path,
                      epoch_bounds=[1, 10],
                      embedding_dim_bounds=[10, 500],
                      rnn_dim_bounds=[10, 500],
                      learning_rate_bounds=[0, 1],
                      momentum_bounds=[0, 1],
                      verbose=True,
                      prefix=""):
    """
    Train model using random params, at each time in search_trials
    the hyper param search is reduce. At the end, the best model
    (with the best accuracy on the valid dataset is select) is returned


    :param model: recurrent model
    :type model: RNN, LSTM, GRU
    :param train_data_path: path to train
    :type train_data_path: str
    :param test_data_path: path to test
    :type test_data_path: str
    :param pkl_path: path to model
    :type pkl_path: str
    :param epochs: number of epochs
    :type epochs: int
    :param embedding_dim: embedding dimention
    :type embedding_dim: int
    :param rnn_dim: rnn hidden size dimention
    :type rnn_dim: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param momentum: momentum param
    :type momentum: float


    :return: test accuracy, hyperperams, pkl path
    :rtype: float, dict, str
    """
    epoch_bounds = epoch_bounds
    embedding_dim_bounds = embedding_dim_bounds
    learning_rate_bounds = learning_rate_bounds
    momentum_bounds = momentum_bounds
    rnn_dim_bounds = rnn_dim_bounds
    best_acc = 0
    best_params = None
    model_path = None

    for i in range(search_trials):
        if verbose:
            print("grid_search ({}/{})\n".format(i + 1, search_trials))

        all_acc, all_hyper_params, all_names = random_search(Model,
                                                             random_trials,
                                                             train_data_path,
                                                             test_data_path,
                                                             epoch_bounds,
                                                             embedding_dim_bounds, # noqa
                                                             rnn_dim_bounds,
                                                             learning_rate_bounds, # noqa
                                                             momentum_bounds,
                                                             verbose=verbose,
                                                             prefix=prefix)
        best_i = np.argmax(all_acc)
        current_acc = all_acc[best_i] # noqa
        current_dict = all_hyper_params[best_i] # noqa
        name = all_names[best_i]
        if best_acc < current_acc:
                epoch_bounds = [epoch_bounds[0],
                                current_dict["epochs"] + 1]
                embedding_dim_bounds = [embedding_dim_bounds[0],
                                        current_dict["embedding_dim"] + 1]
                rnn_dim_bounds = [rnn_dim_bounds[0],
                                  current_dict["rnn_dim"] + 1]
                learning_rate_bounds = [learning_rate_bounds[0],
                                        current_dict["learning_rate"]]
                momentum_bounds = [momentum_bounds[0],
                                   current_dict["momentum"]]
                best_acc = current_acc
                best_params = current_dict
                model_path = name

    test_acc = eval_model_on_test(Model,
                                  train_data_path,
                                  test_data_path,
                                  model_path,
                                  epochs=best_params["epochs"],
                                  embedding_dim=best_params["embedding_dim"], # noqa
                                  rnn_dim=best_params["rnn_dim"], # noqa
                                  learning_rate=best_params["learning_rate"], # noqa
                                  momentum=best_params["momentum"])

    return test_acc, best_params, model_path
