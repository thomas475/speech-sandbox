import torch
import torch.nn.functional as F


class LSTMAutoencoder(torch.nn.Module):
    def __init__(self, learning_rate=1e-3, epochs=10, verbose=1):
        super().__init__()
        self.encoder = self._Encoder()
        self.decoder = self._Decoder()
        self.loss_function = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def fit(self, X):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, X),
            batch_size=64, 
            shuffle=True
        )
        for epoch in range(self.epochs):
            cumulative_loss = 0.0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                X_pred = self(X_batch)
                loss = self.loss_function(X_pred, y_batch)
                loss.backward()
                optimizer.step()
                cumulative_loss += loss.item()
            average_loss = cumulative_loss / len(dataloader)
            if self.verbose >= 1:
                print(f"\rEpoch {epoch + 1}/{self.epochs} - Loss: {average_loss:.4f}", end='', flush=True)
        if self.verbose >= 1:
            print()

    def reconstruct(self, x):
        return self.forward(x)
    
    class _Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
                torch.nn.ReLU()
            )
            self.rnn = torch.nn.LSTM(32, 64, batch_first=True)

        def forward(self, x):
            x = self.cnn(x)
            x = x.permute(0, 2, 1)
            x, _ = self.rnn(x)
            return x
        
    class _Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.LSTM(64, 32, batch_first=True)
            self.cnn = torch.nn.Sequential(
                torch.nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
                torch.nn.Tanh()
            )

        def forward(self, x):
            x, _ = self.rnn(x)
            x = x.permute(0, 2, 1)
            x = self.cnn(x)
            return x
