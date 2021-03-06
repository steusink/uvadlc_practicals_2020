"""
This module implements a bidirectional LSTM in PyTorch.
You should fill in code into indicated sections.
Date: 2020-11-09
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class biLSTM(nn.Module):
    def __init__(
        self,
        seq_length,
        input_dim,
        hidden_dim,
        num_classes,
        batch_size,
        device,
    ):

        super(biLSTM, self).__init__()

        # Initialise hidden state and cell state
        self.h_init = torch.zeros(
            (batch_size, hidden_dim), dtype=torch.float32
        ).to(device)
        self.c_init = torch.zeros(
            (batch_size, hidden_dim), dtype=torch.float32
        ).to(device)

        # Create embedding layer
        self.embed = nn.Embedding(seq_length, input_dim)

        # Create forward and backward cell
        self.fwd = LSTMCell(
            seq_length, input_dim, hidden_dim, num_classes, batch_size, device
        )
        self.bwd = LSTMCell(
            seq_length, input_dim, hidden_dim, num_classes, batch_size, device
        )

        # Last we define the weights and bias for the prediction layer.
        self.p_w = nn.Parameter(torch.empty(num_classes, 2 * hidden_dim))
        self.p_b = nn.Parameter(torch.empty(num_classes))

    def forward(self, x):
        # Create embedding
        embeds = self.embed(x)

        # Perform forward pass
        h_fwd, c_fwd = self.h_init, self.c_init
        # Loop over the embedding matrix and forward each
        # sequence element through the memory cell.
        for i in range(x.size(1)):
            # Get the sequence batch from the ith timestep
            # and reshape it to allign with the shape of h.
            t = embeds[:, i, :]
            c_fwd, h_fwd = self.fwd(t, c_fwd, h_fwd)

        # Perform forward pass
        h_bwd, c_bwd = self.h_init, self.c_init
        # Loop over the embedding matrix and forward each
        # sequence element through the memory cell.
        for i in range(1, x.size(1) + 1):
            # Get the sequence batch from the ith timestep
            # and reshape it to allign with the shape of h.
            t = embeds[:, -i, :]
            c_bwd, h_bwd = self.bwd(t, c_bwd, h_bwd)

        # Concetenate both resulting hidden states
        h_comb = torch.cat((h_fwd, h_bwd), dim=1)

        # Use said hidden state to make predictions
        preds = F.log_softmax(h_comb @ self.p_w.T + self.p_b, dim=1)

        return preds


class LSTMCell(nn.Module):
    def __init__(
        self,
        seq_length,
        input_dim,
        hidden_dim,
        num_classes,
        batch_size,
        device,
    ):

        super(LSTMCell, self).__init__()

        # Create the weight matrices and bias vectors. We will concatenate x
        # and h when forwarding through the memory cell, which is why the
        # second dim is set to input_dim + hidden_dim.
        self.g_w = nn.Parameter(
            torch.empty(hidden_dim, input_dim + hidden_dim)
        )
        self.i_w = nn.Parameter(
            torch.empty(hidden_dim, input_dim + hidden_dim)
        )
        self.f_w = nn.Parameter(
            torch.empty(hidden_dim, input_dim + hidden_dim)
        )
        self.o_w = nn.Parameter(
            torch.empty(hidden_dim, input_dim + hidden_dim)
        )
        self.g_b = nn.Parameter(torch.empty(hidden_dim))
        self.i_b = nn.Parameter(torch.empty(hidden_dim))
        self.f_b = nn.Parameter(torch.empty(hidden_dim))
        self.o_b = nn.Parameter(torch.empty(hidden_dim))

    def forward(self, x, c, h):
        # Concetenate input vector x and hidden layer h
        x_h = torch.cat((x, h), 1)

        # Perform forward pass through all gates by matrix multiplying
        # the weight matrix and x_h vector, adding the bias and applying
        # the corresponding activation function.
        g = torch.tanh(x_h @ self.g_w.T + self.g_b)
        i = torch.sigmoid(x_h @ self.i_w.T + self.i_b)
        f = torch.sigmoid(x_h @ self.f_w.T + self.f_b)
        o = torch.sigmoid(x_h @ self.o_w.T + self.o_b)

        # Compute new cell state and use it to compute
        # the new hidden state.
        new_c = g * i + c * f
        new_h = torch.tanh(new_c) * o

        return new_c, new_h
