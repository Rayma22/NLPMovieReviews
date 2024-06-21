import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from datasets import load_dataset
import pandas as pd

class TextDataset(Dataset):
    """Custom dataset class to handle text data for sentiment analysis or similar tasks."""

    def __init__(self, split_data, tokenizer, vocab, max_length):
        """
        Initializes the dataset with the given data and preprocessing tools.
        
        :param split_data: The dataset split (e.g., train, test) to be processed.
        :param tokenizer: Tokenizer function to convert text to tokens.
        :param vocab: Vocabulary object to convert tokens to indices.
        :param max_length: The maximum length for padding/truncating text sequences.
        """
        self.data = split_data
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        """Returns the number of examples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the tokenized and processed data at the specified index.
        
        :param idx: Index of the data to retrieve.
        :return: A tuple of the tokenized text tensor and the label tensor.
        """
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        tokenized_text = [self.vocab[token] for token in self.tokenizer(text)]
        if len(tokenized_text) > self.max_length:
            tokenized_text = tokenized_text[:self.max_length]
        else:
            tokenized_text += [self.vocab["<pad>"]] * (self.max_length - len(tokenized_text))
        return torch.tensor(tokenized_text), torch.tensor(label)

class TextCNN(nn.Module):
    """A simple text classification model using Convolutional Neural Networks (CNN)."""

    def __init__(self, vocab_size, embed_dim, num_classes):
        """
        Initializes the TextCNN model with the specified vocabulary size, embedding dimension, and number of classes.
        
        :param vocab_size: The size of the vocabulary.
        :param embed_dim: The dimensionality of the word embeddings.
        :param num_classes: The number of output classes for classification.
        """
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv2d(1, 100, (5, embed_dim))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((256 - 5 + 1, 1))
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        :param x: The input tensor containing token indices.
        :return: The output tensor after applying embedding, convolution, pooling, and linear layers.
        """
        x = self.embedding(x).unsqueeze(1)
        x = self.relu(self.conv(x))
        x = self.pool(x).squeeze(3).squeeze(2)
        x = self.fc(x)
        return x

def prepare_data():
    """
    Prepares and returns data loaders for training, validation, and testing along with the vocabulary size and dataset.
    
    :return: Tuple containing data loaders for train, validation, test sets, vocabulary size, and the dataset.
    """
    dataset = load_dataset('rotten_tomatoes')
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for example in data_iter:
            yield tokenizer(example['text'])

    vocab = build_vocab_from_iterator(yield_tokens(dataset['train']), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])

    train_data = TextDataset(dataset['train'], tokenizer, vocab, 256)
    validation_data = TextDataset(dataset['validation'], tokenizer, vocab, 256)
    test_data = TextDataset(dataset['test'], tokenizer, vocab, 256)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=128)
    test_loader = DataLoader(test_data, batch_size=128)

    return train_loader, validation_loader, test_loader, len(vocab), dataset

def train_and_validate(model, train_loader, validation_loader, loss_fn, optimizer, epochs, device):
    """
    Trains and validates the model.
    
    :param model: The model to be trained.
    :param train_loader: DataLoader for the training set.
    :param validation_loader: DataLoader for the validation set.
    :param loss_fn: The loss function.
    :param optimizer: The optimizer.
    :param epochs: The number of training epochs.
    :param device: The device to use for training.
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

def evaluate_test_set(model, test_loader, device):
    """
    Evaluates the model on the test set and outputs the F1 score and prediction results.
    
    :param model: The model to be evaluated.
    :param test_loader: DataLoader for the test set.
    :param device: The device to use for evaluation.
    :return: The macro F1 score for the test set.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_macro_f1 = f1_score(all_labels, all_preds, average='macro')
    results = pd.DataFrame({"index": range(len(all_preds)), "pred": all_preds})
    results.to_csv('results.csv', index=False)
    return test_macro_f1

def main():
    """
    Main function to prepare data, train, validate, and evaluate the TextCNN model.
    """
    train_loader, validation_loader, test_loader, vocab_size, dataset = prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(set(dataset['train']['label']))
    model = TextCNN(vocab_size, 300, num_classes)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_and_validate(model, train_loader, validation_loader, loss_fn, optimizer, 10, device)
    test_macro_f1 = evaluate_test_set(model, test_loader, device)
    print(test_macro_f1)

if __name__ == "__main__":
    main()
