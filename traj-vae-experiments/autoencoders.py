import torch
import torch.nn as nn

class LSTM_MLP(nn.Module):
    # LSTM encoder with MLP decoder
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_MLP, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        output = self.fc(lstm_out[-1])
        return output
