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

import os
import time
from datetime import datetime
import argparse

import pickle
import itertools
import json

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################


def train(config):
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(
        config.batch_size,
        config.seq_length,
        dataset.vocab_size,
        lstm_num_hidden=config.lstm_num_hidden,
        lstm_num_layers=config.lstm_num_layers,
        device=device,
    )
    model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Setup loss and accuracy lists
    loss_list = []
    average_loss = 0
    accuracy_list = []

    for step, (batch_inputs, batch_targets) in enumerate(
        itertools.cycle(data_loader)
    ):

        # Only for time measurement of step through network
        t1 = time.time()

        # Stack batch lists into a tensor instead of a list
        # of tensors.
        batch_inputs = torch.stack(batch_inputs).to(device)
        batch_targets = torch.stack(batch_targets).to(device)

        # Reset model for next iteration
        model.zero_grad()

        # Forward pass. Outputs have shape (seq_length, batch_size, hiddem_dim)
        log_probs_list = model(batch_inputs)

        # Compute the loss, gradients and update the network parameters
        loss = 0
        for t in range(config.seq_length):
            loss += (
                criterion(log_probs_list[t], batch_targets[t])
                / config.seq_length
            )

        # Add to average loss
        average_loss += loss / config.print_every

        # Perform backpropagation
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.max_norm
        )

        optimizer.step()

        # Calculate accuarcy
        accuracy = 0.0
        for t in range(config.seq_length):
            preds = torch.argmax(log_probs_list[t], dim=1)
            correct = (preds == batch_targets[t]).sum().item()
            accuracy += correct / config.batch_size / config.seq_length

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        if (step + 1) % config.print_every == 0:
            print(
                "[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    step,
                    config.train_steps,
                    config.batch_size,
                    examples_per_second,
                    accuracy,
                    loss,
                )
            )
            # Document loss and accuracy
            accuracy_list.append(accuracy)
            loss_list.append(average_loss)
            average_loss = 0

        if (step + 1) % config.sample_every == 0:
            # Save model every time we want to sample new text.
            if config.save_model:
                model_path = "saved_models/step_{}.pickle".format(step)
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error,
            # check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    # Save accuracy, loss, and potentially configuration
    if config.save_config:
        model_path = "saved_models/config.pickle".format(step)
        with open(model_path, "wb") as f:
            pickle.dump(config, f)

    results_dict = {"accuracy": accuracy_list, "loss": loss_list}
    file_path = "results/results_dict.pickle"
    with open(file_path, "wb") as f:
        pickle.dump(results_dict, f)

    print("Done training.")


###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument(
        "--txt_file",
        type=str,
        required=True,
        help="Path to a .txt file to train on",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=30,
        help="Length of an input sequence",
    )
    parser.add_argument(
        "--lstm_num_hidden",
        type=int,
        default=256,
        help="Number of hidden units in the LSTM",
    )
    parser.add_argument(
        "--lstm_num_layers",
        type=int,
        default=2,
        help="Number of LSTM layers in the model",
    )

    # Training params
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of examples to process in a batch",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-3, help="Learning rate"
    )

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument(
        "--learning_rate_decay",
        type=float,
        default=0.96,
        help="Learning rate decay fraction",
    )
    parser.add_argument(
        "--learning_rate_step",
        type=int,
        default=5000,
        help="Learning rate step",
    )
    parser.add_argument(
        "--dropout_keep_prob",
        type=float,
        default=1.0,
        help="Dropout keep probability",
    )

    parser.add_argument(
        "--train_steps",
        type=int,
        default=int(1e6),
        help="Number of training steps",
    )
    parser.add_argument("--max_norm", type=float, default=5.0, help="--")

    # Misc params
    parser.add_argument(
        "--summary_path",
        type=str,
        default="./summaries/",
        help="Output path for summaries",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=50,
        help="How often to print training progress",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=100,
        help="How often to sample from the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cpu" if not torch.cuda.is_available() else "cuda"),
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--save_model",
        type=bool,
        default=False,
        help="Select whether the model needs to be saved for every sample iteration",
    )
    parser.add_argument(
        "--save_config",
        type=bool,
        default=False,
        help="Select whether the configuration needs to be saved for every sample iteration",
    )

    config = parser.parse_args()

    # Train the model
    train(config)
