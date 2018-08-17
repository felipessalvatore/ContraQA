from util import train_in_epoch, get_valid_loss
from util import training_loop_text_classification
from util import pre_process_df
from RNN import RNN
from DataHolder import DataHolder
from RNNConfig import RNNConfig
import torch
from torchtext import data
import pandas as pd
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

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

train_data, test_data = pre_process_df(train_data, test_data)

train_data_path = os.path.join(parentdir,
                               "text_generation",
                               "data",
                               "boolean1_train_processed.csv")

test_data_path = os.path.join(parentdir,
                              "text_generation",
                              "data",
                              "boolean1_test_processed.csv")

train_data.to_csv(train_data_path, header=False, index=False)
test_data.to_csv(test_data_path, header=False, index=False)


# TEXT = data.Field()
TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(tensor_type=torch.FloatTensor)

train = data.TabularDataset(path=train_data_path,
                            format="csv",
                            fields=[('text', TEXT), ('label', LABEL)])

test = data.TabularDataset(path=test_data_path,
                           format="csv",
                           fields=[('text', TEXT), ('label', LABEL)])

train, valid = train.split(0.8)


TEXT.build_vocab(train, max_size=25000)
LABEL.build_vocab(train)


config = RNNConfig(vocab_size=len(TEXT.vocab),
                   output_dim=len(LABEL.vocab))

data = DataHolder(config, train, valid, test)


model = RNN(config)


training_loop_text_classification(model,
                                  config,
                                  data,
                                  "rnn_boolean1.pkl")

labels_legend = ['no', 'yes']
test_bach = next(iter(data.test_iter))
_, pred, labels = model.evaluate_bach(test_bach)

plot_confusion_matrix(truth=labels.numpy(),
                      predictions=pred.numpy(),
                      save=True,
                      path="rnn_confusion_matrix_boolean1.png",
                      classes=labels_legend)
