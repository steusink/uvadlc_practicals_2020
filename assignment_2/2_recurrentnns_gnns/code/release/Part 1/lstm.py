"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch


class LSTM(nn.Module):
    def __init__(
        self,
        seq_length,
        input_dim,
        hidden_dim,
        num_classes,
        batch_size,
        device,
    ):
        super(LSTM, self).__init__()

        # Initialise hidden state and cell state
        self.h_init = torch.zeros(
            (batch_size, hidden_dim), dtype=torch.float32
        ).to(device)
        self.c_init = torch.zeros(
            (batch_size, hidden_dim), dtype=torch.float32
        ).to(device)

        # Create embedding layer
        self.embed = nn.Embedding(seq_length, input_dim)

        # Create the weight matrices and bias vectors. We will concatenate x
        # and h when forwarding through the memory cell, which is why the
        # second dim is set to input_dim + hidden_dim.
        self.w_g = nn.Parameter(
            torch.empty(hidden_dim, input_dim + hidden_dim)
        )
        self.w_i = nn.Parameter(
            torch.empty(hidden_dim, input_dim + hidden_dim)
        )
        self.w_f = nn.Parameter(
            torch.empty(hidden_dim, input_dim + hidden_dim)
        )
        self.w_o = nn.Parameter(
            torch.empty(hidden_dim, input_dim + hidden_dim)
        )
        self.b_g = nn.Parameter(torch.empty(hidden_dim))
        self.b_i = nn.Parameter(torch.empty(hidden_dim))
        self.b_f = nn.Parameter(torch.empty(hidden_dim))
        self.b_o = nn.Parameter(torch.empty(hidden_dim))

        # Last we define the weights and bias for the prediction layer.
        self.w_p = nn.Parameter(torch.empty(num_classes, hidden_dim))
        self.b_p = nn.Parameter(torch.empty(num_classes))

    def cell_forward(self, x, c, h):
        # Concetenate input vector x and hidden layer h
        x_h = torch.cat((x, h), 1)

        # Perform forward pass through all gates by matrix multiplying
        # the weight matrix and x_h vector, adding the bias and applying
        # the corresponding activation function.
        g = torch.tanh(x_h @ self.w_g.T + self.b_g)
        i = torch.sigmoid(x_h @ self.w_i.T + self.b_i)
        f = torch.sigmoid(x_h @ self.w_f.T + self.b_f)
        o = torch.sigmoid(x_h @ self.w_o.T + self.b_o)

        # Compute new cell state and use it to compute
        # the new hidden state.
        new_c = g * i + c * f
        new_h = torch.tanh(new_c) * o

        return new_c, new_h

    def forward(self, x):
        # Create embedding. This will be a matrix of size
        # seq_length * num_inputs
        embeds = self.embed(x)

        # Set hidden and cell state to the initial zero
        # vectors.
        h, c = self.h_init, self.c_init
        # Loop over the embedding matrix and forward each
        # sequence element through the memory cell.
        for i in range(x.size(1)):
            # Get the sequence batch from the ith timestep
            # and reshape it to allign with the shape of h.
            t = embeds[:, i, :]
            h, c = self.cell_forward(t, h, c)

        # Use the final hidden state for prediction
        preds = F.softmax(h @ self.w_p.T + self.b_p, dim=1)

        return preds
