import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import math


def GRUCellExposed(input_tensor, hidden, w_ih, w_hh, b_ih=None, b_hh=None):

    gi = F.linear(input_tensor, w_ih, b_ih)
    gh = F.linear(hidden, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = torch.sigmoid(i_r + h_r)
    updategate = torch.sigmoid(i_i + h_i)
    newgate = torch.tanh(i_n + resetgate * h_n)
    hy = (updategate * newgate) + ((1 - updategate) * hidden)

    return hy, resetgate, updategate, newgate


class GRUCell(nn.modules.rnn.RNNCellBase):
    r"""A gated recurrent unit (GRU) cell

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    where :math:`\sigma` is the sigmoid function.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: `True`

    Inputs: input, hidden
        - **input** of shape `(batch, input_size)`: tensor containing input features # noqa
        - **hidden** of shape `(batch, hidden_size)`: tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size x input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`

    Examples::

        >>> rnn = nn.GRUCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx = rnn(input[i], hx)
                output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx)
        hy, resetgate, updategate, newgate = GRUCellExposed(input,
                                                            hx,
                                                            self.weight_ih,
                                                            self.weight_hh,
                                                            self.bias_ih,
                                                            self.bias_hh)
        return hy, resetgate, updategate, newgate


class GRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.GRU_cell = GRUCell(config.embedding_dim, config.rnn_dim)
        self.fc = nn.Linear(config.rnn_dim, config.output_dim)

    def forward(self, x):
        """
        Apply the model to the input x

        :param x: indices of the sentence
        :type x: torch.Tensor(shape=[sent len, batch size]
                              dtype=torch.int64)
        """
        embedded = self.embedding(x)
        hx = torch.zeros((x.shape[1], self.config.rnn_dim))
        all_hidden = []
        reset_gates = []
        update_gates = []
        new_gates = []
        for i in range(x.shape[0]):
            hx, r, u, new = self.GRU_cell(embedded[i], hx)
            all_hidden.append(hx)
            reset_gates.append(r)
            update_gates.append(u)
            new_gates.append(new)

        all_hidden = torch.stack(all_hidden)
        reset_gates = torch.stack(reset_gates)
        update_gates = torch.stack(update_gates)
        new_gates = torch.stack(new_gates)

        self.output = all_hidden
        hidden = all_hidden[-1, :, :]
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        self.reset_gates = reset_gates
        self.update_gates = update_gates
        self.new_gates = new_gates

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
