import numpy as np
import torch
import time
import unittest
import os
import shutil
from contra_qa.text_generation.boolean1_neg import boolean1
from contra_qa.train_functions.util import get_data, training_loop_text_classification # noqa
from contra_qa.train_functions.RNNConfig import RNNConfig
from contra_qa.train_functions.RNN import RNN
from contra_qa.train_functions.LSTM import LSTM
from contra_qa.train_functions.GRU import GRU
from contra_qa.train_functions.DataHolder import DataHolder
from contra_qa.plots.functions import plot_confusion_matrix
from contra_qa.train_functions.random_search import train_model_on_params
from contra_qa.train_functions.random_search import random_search
from contra_qa.train_functions.random_search import naive_grid_search


class TrainFunctionsTest(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        if os.path.exists("data"):
            shutil.rmtree("data")
        if os.path.exists("tmp_pkl"):
            shutil.rmtree("tmp_pkl")
        if os.path.exists("testRNN.pkl"):
            os.remove("testRNN.pkl")
        if os.path.exists("testRNN.png"):
            os.remove("testRNN.png")
        if os.path.exists("testLSTM.pkl"):
            os.remove("testLSTM.pkl")
        if os.path.exists("testLSTM.png"):
            os.remove("testLSTM.png")
        if os.path.exists("testGRU.pkl"):
            os.remove("testGRU.pkl")
        if os.path.exists("testGRU.png"):
            os.remove("testGRU.png")

    @classmethod
    def setUp(cls):
        boolean1()
        cls.path_train = os.path.join("data",
                                      "boolean1_train.csv")
        cls.path_test = os.path.join("data",
                                     "boolean1_test.csv")
        TEXT, LABEL, train, valid, test = get_data(cls.path_train,
                                                   cls.path_test)

        cls.current_config = RNNConfig(vocab_size=len(TEXT.vocab),
                                       output_dim=len(LABEL.vocab),
                                       epochs=5,
                                       embedding_dim=100,
                                       learning_rate=0.05,
                                       momentum=0.1)

        cls.current_data = DataHolder(cls.current_config,
                                      train,
                                      valid,
                                      test)

    def test_basic_training_RNN(self):
        model = RNN(self.current_config)

        training_loop_text_classification(model,
                                          self.current_config,
                                          self.current_data,
                                          "testRNN.pkl",
                                          verbose=False)
        model = RNN(self.current_config)
        valid_batch = next(iter(self.current_data.valid_iter))
        model.load_state_dict(torch.load("testRNN.pkl"))
        acc, pred, labels = model.evaluate_bach(valid_batch)
        self.assertTrue(acc > 0.7,
                        "after training, valid_acc = {:.3f}".format(acc))
        msg = "problems with the confusion matrix plot"
        labels_legend = ['no', 'yes']
        plot_confusion_matrix(truth=labels.numpy(),
                              predictions=pred.numpy(),
                              save=True,
                              path="testRNN.png",
                              classes=labels_legend)
        self.assertTrue(os.path.exists("testRNN.png"), msg=msg)

    def test_basic_training_LSTM(self):
        model = LSTM(self.current_config)
        training_loop_text_classification(model,
                                          self.current_config,
                                          self.current_data,
                                          "testLSTM.pkl",
                                          verbose=False)
        model = LSTM(self.current_config)
        valid_batch = next(iter(self.current_data.valid_iter))
        model.load_state_dict(torch.load("testLSTM.pkl"))
        valid_batch = next(iter(self.current_data.valid_iter))
        model.load_state_dict(torch.load("testLSTM.pkl"))
        acc, pred, labels = model.evaluate_bach(valid_batch)
        self.assertTrue(acc > 0.7,
                        "after training, valid_acc = {:.3f}".format(acc))
        msg = "problems with the confusion matrix plot"
        labels_legend = ['no', 'yes']
        plot_confusion_matrix(truth=labels.numpy(),
                              predictions=pred.numpy(),
                              save=True,
                              path="testLSTM.png",
                              classes=labels_legend)
        self.assertTrue(os.path.exists("testLSTM.png"), msg=msg)

    def test_basic_training_GRU(self):
        model = GRU(self.current_config)
        training_loop_text_classification(model,
                                          self.current_config,
                                          self.current_data,
                                          "testGRU.pkl",
                                          verbose=False)
        model = GRU(self.current_config)
        valid_batch = next(iter(self.current_data.valid_iter))
        model.load_state_dict(torch.load("testGRU.pkl"))
        valid_batch = next(iter(self.current_data.valid_iter))
        model.load_state_dict(torch.load("testGRU.pkl"))
        acc, pred, labels = model.evaluate_bach(valid_batch)
        self.assertTrue(acc > 0.7,
                        "after training, valid_acc = {:.3f}".format(acc))
        msg = "problems with the confusion matrix plot"
        labels_legend = ['no', 'yes']
        plot_confusion_matrix(truth=labels.numpy(),
                              predictions=pred.numpy(),
                              save=True,
                              path="testGRU.png",
                              classes=labels_legend)
        self.assertTrue(os.path.exists("testGRU.png"), msg=msg)

    def test_random_param_train(self):

        acc1 = train_model_on_params(RNN,
                                     self.path_train,
                                     self.path_test,
                                     "testRNN.png",
                                     5,
                                     100,
                                     100,
                                     0.05,
                                     0.1)

        acc2 = train_model_on_params(GRU,
                                     self.path_train,
                                     self.path_test,
                                     "testRNN.png",
                                     5,
                                     50,
                                     50,
                                     0.05,
                                     0.1)

        acc3 = train_model_on_params(LSTM,
                                     self.path_train,
                                     self.path_test,
                                     "testRNN.png",
                                     5,
                                     23,
                                     30,
                                     0.05,
                                     0.1)
        acc = acc1 + acc2 + acc3
        msg = "after training, valid_acc = {:.3f}".format(acc)
        self.assertTrue(acc >= 0.6 * 3, msg=msg)

    def test_random_param_train_bound(self):

        all_acc, all_hyper_params, all_names = random_search(RNN,
                                                             10,
                                                             self.path_train,
                                                             self.path_test,
                                                             epoch_bounds=[1, 2], # noqa
                                                             embedding_dim_bounds=[10, 500], # noqa
                                                             rnn_dim_bounds=[10, 500], # noqa
                                                             learning_rate_bounds=[0, 1], # noqa
                                                             momentum_bounds=[0, 1], # noqa
                                                             verbose=False, # noqa
                                                             prefix="RNN_boolean1_", # noqa
                                                             acc_bound=0.6) # noqa
        print(all_acc, all_hyper_params, all_names)
        cond = len(all_acc) == len(all_hyper_params) == len(all_names)
        cond_bound = len(all_acc) < 10
        self.assertTrue(cond, msg="different output sizes")
        self.assertTrue(cond_bound,
                        msg="not stoping, len(all_acc) = {}".format(len(all_acc))) # noqa

    def test_grid_search_bound(self):

        init = time.time()
        _, _, _ = naive_grid_search(RNN,
                                    1,
                                    1,
                                    self.path_train,
                                    self.path_test,
                                    epoch_bounds=[1, 2],
                                    verbose=False,
                                    prefix="RNN_boolean1_")
        reference = time.time() - init

        init = time.time()

        test_accRNN, _, _ = naive_grid_search(RNN,
                                              10,
                                              10,
                                              self.path_train,
                                              self.path_test,
                                              epoch_bounds=[1, 2],
                                              verbose=False,
                                              prefix="RNN_boolean1_",
                                              acc_bound=0.5)
        experiment = time.time() - init
        msg = "taking too much time, ref ={:.3f}, exp ={:.3f}".format(reference, experiment)# noqa
        cond = experiment <= 3 * reference
        self.assertTrue(cond, msg=msg)

    def test_random_search(self):
        all_acc, all_hyper_params, all_names = random_search(RNN,
                                                             2,
                                                             self.path_train,
                                                             self.path_test,
                                                             epoch_bounds=[1, 2], # noqa
                                                             embedding_dim_bounds=[10, 500], # noqa
                                                             rnn_dim_bounds=[10, 500], # noqa
                                                             learning_rate_bounds=[0, 1], # noqa
                                                             momentum_bounds=[0, 1], # noqa
                                                             verbose=False, # noqa
                                                             prefix="RNN_boolean1_") # noqa
        cond = len(all_acc) == len(all_hyper_params) == len(all_names)
        self.assertTrue(cond, msg="different output sizes")
        self.assertTrue(np.max(all_acc) > 0.56,
                        msg="acc list = {}".format(all_acc))

    def test_naive_grid_search(self):
        test_accRNN, _, _ = naive_grid_search(RNN,
                                              2,
                                              2,
                                              self.path_train,
                                              self.path_test,
                                              epoch_bounds=[1, 2],
                                              verbose=False,
                                              prefix="RNN_boolean1_")
        test_accGRU, _, _ = naive_grid_search(GRU,
                                              2,
                                              2,
                                              self.path_train,
                                              self.path_test,
                                              epoch_bounds=[1, 2],
                                              verbose=False,
                                              prefix="GRU_boolean1_")

        test_accLSTM, _, _ = naive_grid_search(LSTM,
                                               2,
                                               2,
                                               self.path_train,
                                               self.path_test,
                                               epoch_bounds=[1, 2],
                                               verbose=False,
                                               prefix="LSTM_boolean1_")
        acc = test_accRNN + test_accGRU + test_accLSTM
        msg = "after training, valid_acc = {:.3f}, {:.3f}, {:.3f}".format(test_accRNN, # noqa
                                                                          test_accGRU, # noqa
                                                                          test_accLSTM) # noqa
        self.assertTrue(acc >= 0.6 * 3, msg=msg)

