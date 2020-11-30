import pickle
import torch
import random
from dataset import TextDataset


def sample_sentence(model, dataset, device, seq_length):
    # Initialise generated sequence by randomly generating
    # a character from the vocabulary
    int_seq = [random.randint(0, dataset.vocab_size - 1)]

    # Keep running the sentence through the model, appending
    # the most likely character each time until the desired
    # length is reached.
    while len(int_seq) <= seq_length:
        input_tensor = torch.LongTensor(int_seq).to(device)
        input_tensor = input_tensor.reshape(len(int_seq), 1)
        with torch.no_grad():
            log_probs_list = model(input_tensor)
        next_int = int(torch.argmax(log_probs_list[-1]))
        int_seq.append(next_int)

    # Convert to string:
    string = dataset.convert_to_string(int_seq)

    return string


def sample_multiple_sentences(models, lengths, dataset, device):
    # Set random seed for integer gerneration
    random.seed(42)
    i = 1
    for model in models:
        print("Model {}/3".format(i))
        i += 1
        for length in lengths:
            print("Sequences of length ", length)
            for j in range(1, 6):
                print("Sample {}".format(j))
                print(sample_sentence(model, dataset, device, length))


def sample_sentence_random(model, temp, dataset, device, seq_length):
    # Initialise generated sequence by randomly generating
    # a character from the vocabulary
    int_seq = [random.randint(0, dataset.vocab_size - 1)]

    # Keep running the sentence through the model, appending
    # the most likely character each time until the desired
    # length is reached.
    while len(int_seq) <= seq_length:
        input_tensor = torch.LongTensor(int_seq).to(device)
        input_tensor = input_tensor.reshape(len(int_seq), 1)
        with torch.no_grad():
            log_probs = model(input_tensor, temp=temp)[-1]

        # Convert back to number between 0 and 1
        probs = torch.exp(log_probs)
        next_int = torch.multinomial(probs, 1).squeeze()
        int_seq.append(int(next_int))

    # Convert to string:
    string = dataset.convert_to_string(int_seq)

    return string


if __name__ == "__main__":

    # Load models and config
    with open("saved_models/step_16599.pickle", "rb") as f:
        model1 = pickle.load(f)
    with open("saved_models/step_33199.pickle", "rb") as f:
        model2 = pickle.load(f)
    with open("saved_models/step_49799.pickle", "rb") as f:
        model3 = pickle.load(f)

    models = [model1, model2, model3]

    with open("saved_models/config.pickle", "rb") as f:
        config = pickle.load(f)

    # Load dataset and device used for the model
    dataset = TextDataset(config.txt_file, config.seq_length)
    device = config.device
    print("device =", device)

    # Define sequence lengths
    lengths = [15, 30, 45]

    # Sample descretely
    sample_multiple_sentences(models, lengths, dataset, device)

    # Sample randomly
    temps = [0.5, 1.0, 2.0]

    print("Start Random Sampling")
    for temp in temps:
        print("temp = ", temp)
        sentence = sample_sentence_random(model3, temp, dataset, device, 45)
        print(sentence)
