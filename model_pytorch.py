import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import torch
import torch.nn as nn
import torch.optim as optim


data = pd.read_csv('./data/preprocessed/ACPs_Breast_cancer.csv')
data['sequence'] = data['sequence'].map(lambda x: [str(y) for y in x])  # Convert to lists of characters

word_dictionary = {y: x for x, y in enumerate(list(set(data['sequence'].sum())))}
data['sequence'] = data['sequence'].map(lambda x: [word_dictionary[y] for y in x])
max_sequence_length = max(data['sequence'].apply(len))


y = data.pop('class')
X = data

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values.reshape(-1, 1), test_size=0.4, random_state=1, stratify=y)

X_train = sequence.pad_sequences(np.squeeze(X_train), maxlen=max_sequence_length)
X_test = sequence.pad_sequences(np.squeeze(X_test), maxlen=max_sequence_length)

train_input = torch.from_numpy(X_train)
train_target = torch.from_numpy(y_train).type(torch.FloatTensor)
test_input = torch.from_numpy(X_test)
test_target = torch.from_numpy(y_test)


class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_rate):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True)

    def forward(self, x):
        # x shape: (seq_length, N) N = batch_size
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        output, (hidden, cell) = self.rnn(embedding)
        return hidden


class Model(nn.Module):

    def __init__(self, encoder: Encoder, hidden_size):
        super(Model, self).__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out = self.encoder(x)
        pred = self.linear(lstm_out)[-1]
        return torch.sigmoid(pred).view(-1)


num_epochs = 200
learning_rate = 0.001
batch_size = 64
top_words = len(word_dictionary)
embedding_vector_length = 32
lstm_hidden_size = 100
lstm_num_layers = 1
dropout_rate = 0

encoder_net = Encoder(max_sequence_length, embedding_vector_length, lstm_hidden_size, lstm_num_layers, dropout_rate)
# Define model
model = Model(encoder_net, hidden_size=lstm_hidden_size)
# Define optimiser
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# Define loss function
loss_fn = nn.BCELoss()  # (Binary Cross Entropy assuming output has already gone through sigmoid)


# Loop through all epochs
for epoch in range(num_epochs):
    # Forward pass
    scores = model(train_input)
    # Define loss
    loss = loss_fn(scores.view(-1, 1), train_target)
    print("Epoch:", epoch, "Loss:", loss)
    # For every mini-batch during training we need to explicitly set the gradients to zero
    # before backpropagation because PyTorch accumulates gradients on subsequent backward passes
    optimizer.zero_grad()
    # Backpropagation
    loss.backward()
    # Perform parameter update based on the current gradient
    optimizer.step()


model.eval()

predictions = model(test_input).detach().numpy().reshape(-1, 1)

predictions = np.where(predictions > 0.5, 1, 0)
targets = y_test

np.concatenate((predictions, targets), axis=1)

print(classification_report(targets, predictions))