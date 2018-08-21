import torch
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


class TrainFunctionsTest(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        if os.path.exists("data"):
            shutil.rmtree("data")
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

    def test_random_function(self):

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
