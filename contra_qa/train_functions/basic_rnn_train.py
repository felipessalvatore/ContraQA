from util import training_loop_text_classification, get_data
from RNN import RNN
from DataHolder import DataHolder
from RNNConfig import RNNConfig
import os
import inspect
import sys

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from plots.functions import plot_confusion_matrix, plot_histogram_from_labels # noqa
from text_processing.functions import simple_pre_process_text_df # noqa


train_data_path = os.path.join(parentdir,
                               "text_generation",
                               "data",
                               "boolean1_train.csv")

test_data_path = os.path.join(parentdir,
                              "text_generation",
                              "data",
                              "boolean1_test.csv")

data_folder = os.path.join(parentdir,
                           "text_generation",
                           "data")

TEXT, LABEL, train, valid, test = get_data(train_data_path,
                                           test_data_path,
                                           data_folder)


current_config = RNNConfig(vocab_size=len(TEXT.vocab),
                           output_dim=len(LABEL.vocab),
                           epochs=3,
                           embedding_dim=100,
                           learning_rate=0.01,
                           momentum=0.2)

current_data = DataHolder(current_config,
                          train,
                          valid,
                          test)

model = RNN(current_config)

training_loop_text_classification(model,
                                  current_config,
                                  current_data,
                                  "rnn_boolean1.pkl")

labels_legend = ['no', 'yes']
test_bach = next(iter(current_data.test_iter))
acc, pred, labels = model.evaluate_bach(test_bach)

plot_confusion_matrix(truth=labels.numpy(),
                      predictions=pred.numpy(),
                      save=True,
                      path="rnn_confusion_matrix_boolean1.png",
                      classes=labels_legend)

assert acc > 0.7
