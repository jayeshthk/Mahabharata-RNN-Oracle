# Mahabharata-RNN-Oracle

**Mahabharata-RNN-Oracle** is a Recurrent Neural Network (RNN) model designed to predict the next character in sequences from the epic Mahabharata. This project utilizes Andréj Karpathy's RNN tutorial as a basis for developing a language model tailored to one of the most complex and influential texts in Indian literature.

## Overview

This repository includes:

- Data preprocessing scripts to encode the Mahabharata text.
- Implementation of an RNN model with LSTM cells for sequence prediction.
- Training pipeline to fit the model on the Mahabharata dataset.
- Text generation functionality to produce sequences based on the trained model.

## Features

- **Data Encoding**: Converts text into numerical format and one-hot encoded tensors.
- **RNN Model**: Defines a character-level RNN using LSTM layers for sequence modeling.
- **Training and Evaluation**: Trains the model with specified hyperparameters and evaluates its performance.
- **Text Generation**: Generates text based on a seed string using the trained model.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- PyTorch

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/Mahabharata-RNN-Oracle.git
   cd Mahabharata-RNN-Oracle
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch numpy
   ```

### Data Preparation

The script downloads and preprocesses the Mahabharata text data:

1. **Download the Data**:

   ```python
   !wget https://raw.githubusercontent.com/kunjee17/mahabharata/master/books/14.txt
   ```

2. **Read and Encode the Text**:
   The text file is read and encoded into numerical values and one-hot tensors.

### Training the Model

1. **Define and Initialize the Model**:

   ```python
   from model import CharRNN

   n_hidden = 512
   n_layers = 2
   net = CharRNN(chars, n_hidden, n_layers)
   ```

2. **Train the Model**:

   ```python
   from train import train

   train(net, encoded, epochs=20, batch_size=128, seq_length=100, lr=0.001, print_every=10)
   ```

3. **Save the Model**:
   ```python
   model_name = 'rnn_20_epoch.net'
   checkpoint = {'n_hidden': net.n_hidden,
                 'n_layers': net.n_layers,
                 'state_dict': net.state_dict(),
                 'tokens': net.chars}
   with open(model_name, 'wb') as f:
       torch.save(checkpoint, f)
   ```

### Generating Text

To generate text using the trained model:

1. **Load the Model**:

   ```python
   with open('rnn_20_epoch.net', 'rb') as f:
       checkpoint = torch.load(f)

   loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
   loaded.load_state_dict(checkpoint['state_dict'])
   ```

2. **Sample Text**:

   ```python
   from generate import sample

   print(sample(loaded, 1000, prime='krishna was saying', top_k=5))
   ```

## Directory Structure

- `data_preparation.py`: Script for downloading and preparing the Mahabharata text data.
- `model.py`: Defines the `CharRNN` class and model architecture.
- `train.py`: Contains the training loop and validation logic.
- `generate.py`: Functions for text generation using the trained model.

## Contributing

Contributions are welcome! If you have improvements, bug fixes, or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to adjust the directory structure and script names based on your actual project organization.

```

This `README.md` provides a clear and structured overview of your project, making it easier for users to understand, install, and use your repository. Adjust the content according to any additional specifics or changes in your project.
```
