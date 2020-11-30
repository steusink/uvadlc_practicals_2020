# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGenerationModel(nn.Module):
    def __init__(
        self,
        batch_size,
        seq_length,
        vocabulary_size,
        lstm_num_hidden=256,
        lstm_num_layers=2,
        device="cuda:0",
    ):

        super(TextGenerationModel, self).__init__()

        # Set imbed dim to one fourth of hiddem dim.
        embed_dim = int(round(lstm_num_hidden / 4))

        # Initialise embedding
        self.embed = nn.Embedding(vocabulary_size, embed_dim)

        # Initialise LSTM cells
        self.cells = nn.LSTM(embed_dim, lstm_num_hidden, lstm_num_layers)

        # Initialise linear layer
        self.w = nn.Parameter(torch.empty(vocabulary_size, lstm_num_hidden))
        self.b = nn.Parameter(torch.empty(vocabulary_size))

        # Flatten parameters
        self.cells.flatten_parameters()

    def forward(self, x, temp=1):
        # Create embedding
        embeds = self.embed(x)

        # Send input through LSTM
        out, _ = self.cells(embeds)

        # Initialise output list
        log_prob_list = []

        # Loop over all timesteps and compute the output vector.
        for h_t in out:
            p_t = h_t @ self.w.T + self.b
            log_prob_t = F.log_softmax(temp * p_t, dim=1)
            log_prob_list.append(log_prob_t)

        return log_prob_list
