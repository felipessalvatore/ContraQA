import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import math


def LSTMCellExposed(input_tensor,
                    hidden, w_ih, w_hh, b_ih=None, b_hh=None):

    hx, cx = hidden
    gates = F.linear(input_tensor, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy, ingate, forgetgate, cellgate, outgate


class LSTMCell(nn.modules.rnn.RNNCellBase):
    r"""A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_size)`: tensor containing input features # noqa
        - **h_0** of shape `(batch, hidden_size)`: tensor containing the initial hidden # noqa
          state for each element in the batch.
        - **c_0** of shape `(batch, hidden_size)`: tensor containing the initial cell state # noqa
          for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

    Outputs: h_1, c_1
        - **h_1** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch
        - **c_1** of shape `(batch, hidden_size)`: tensor containing the next cell state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size x input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`

    Examples::

        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx, cx = rnn(input[i], (hx, cx))
                output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0),
                                 self.hidden_size,
                                 requires_grad=False)
            hx = (hx, hx)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        hy, cy, ingate, forgetgate, cellgate, outgate = LSTMCellExposed(input,
                                                                        hx,
                                                                        self.weight_ih, # noqa
                                                                        self.weight_hh, # noqa
                                                                        self.bias_ih, # noqa
                                                                        self.bias_hh) # noqa
        return hy, cy, ingate, forgetgate, cellgate, outgate


class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.vocab_size,
                                      self.config.embedding_dim)
        self.LSTM_cell = LSTMCell(self.config.embedding_dim,
                                  self.config.rnn_dim)
        self.fc = nn.Linear(self.config.rnn_dim,
                            self.config.output_dim)

    def forward(self, x):
        """
        Apply the model to the input x

        :param x: indices of the sentence
        :type x: torch.Tensor(shape=[sent len, batch size]
                              dtype=torch.int64)
        """
        embedded = self.embedding(x)
        hx = torch.zeros((x.shape[1],
                          self.config.rnn_dim))
        cx = torch.zeros((x.shape[1],
                          self.config.rnn_dim))
        all_hidden = []
        in_gates = []
        forget_gates = []
        cell_gates = []
        out_gates = []
        for i in range(x.shape[0]):
            hx, cx, ingate, forgetgate, cellgate, outgate = self.LSTM_cell(embedded[i], (hx, cx)) # noqa
            all_hidden.append(hx)
            in_gates.append(ingate)
            forget_gates.append(forgetgate)
            cell_gates.append(cellgate)
            out_gates.append(outgate)

        all_hidden = torch.stack(all_hidden)
        in_gates = torch.stack(in_gates)
        forget_gates = torch.stack(forget_gates)
        cell_gates = torch.stack(cell_gates)
        out_gates = torch.stack(out_gates)

        self.output = all_hidden
        hidden = all_hidden[-1, :, :]

        self.in_gates = in_gates
        self.forget_gates = forget_gates
        self.cell_gates = cell_gates
        self.out_gates = out_gates

        out = self.fc(hidden)
        return out

    def predict(self, x):
        out = self.forward(x)
        softmax = nn.Softmax(dim=1)
        out = softmax(out)
        indices = torch.argmax(out, 1)
        return indices

    def evaluate_bach(self, batch):
        prediction = self.predict(batch.text)
        labels = batch.label.type('torch.LongTensor')
        correct = torch.sum(torch.eq(prediction, labels)).float()
        accuracy = float(correct / labels.shape[0])
        return accuracy, prediction, labels
