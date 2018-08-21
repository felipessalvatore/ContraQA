import torch
import unittest
import os
import shutil
from contra_qa.text_generation.boolean1_neg import boolean1
from contra_qa.train_functions.util import get_data, training_loop_text_classification # noqa
from contra_qa.train_functions.RNNConfig import RNNConfig
from contra_qa.train_functions.RNN import RNN
from contra_qa.train_functions.DataHolder import DataHolder
from contra_qa.plots.functions import plot_confusion_matrix


class TrainFunctionsTest(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        if os.path.exists("data"):
            shutil.rmtree("data")
        if os.path.exists("test.pkl"):
            os.remove("test.pkl")
        if os.path.exists("test.png"):
            os.remove("test.png")

    @classmethod
    def setUp(cls):
        boolean1()
        path_train = os.path.join("data",
                                  "boolean1_train.csv")
        path_test = os.path.join("data",
                                 "boolean1_test.csv")
        TEXT, LABEL, train, valid, test = get_data(path_train,
                                                   path_test)

        cls.current_config = RNNConfig(vocab_size=len(TEXT.vocab),
                                       output_dim=len(LABEL.vocab),
                                       epochs=3,
                                       embedding_dim=100,
                                       learning_rate=0.01,
                                       momentum=0.2)

        cls.current_data = DataHolder(cls.current_config,
                                      train,
                                      valid,
                                      test)

        model = RNN(cls.current_config)

        training_loop_text_classification(model,
                                          cls.current_config,
                                          cls.current_data,
                                          "test.pkl",
                                          verbose=False)

    def test_basic_training_acc(self):
        model = RNN(self.current_config)
        valid_batch = next(iter(self.current_data.valid_iter))
        model.load_state_dict(torch.load("test.pkl"))
        acc, pred, labels = model.evaluate_bach(valid_batch)
        self.assertTrue(acc > 0.7,
                        "after training, valid_acc = {:.3f}".format(acc))
        msg = "problems with the confusion matrix plot"
        labels_legend = ['no', 'yes']
        plot_confusion_matrix(truth=labels.numpy(),
                              predictions=pred.numpy(),
                              save=True,
                              path="test.png",
                              classes=labels_legend)
        self.assertTrue(os.path.exists("test.png"), msg=msg)
