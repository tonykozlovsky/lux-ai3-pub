from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score

B = 32

class ResidualBlockOld(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ResidualBlockOld, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)  # Save the input for the skip connection
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)  # Apply ReLU after BatchNorm
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual  # Add skip connection
        x = nn.ReLU()(x)  # Final ReLU activation
        return x



class SELayer(nn.Module):
    def __init__(self, n_channels: int, reduction: int = 16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        y = self.fc(y.view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            normalize: bool = False,
            activation: Callable = nn.ReLU,
            squeeze_excitation: bool = True,
            **conv2d_kwargs
    ):
        super(ResidualBlock, self).__init__()

        # Calculate "same" padding
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # https://www.wolframalpha.com/input/?i=i%3D%28i%2B2x-k-%28k-1%29%28d-1%29%2Fs%29+%2B+1&assumption=%22i%22+-%3E+%22Variable%22
        assert "padding" not in conv2d_kwargs.keys()
        k = kernel_size
        d = conv2d_kwargs.get("dilation", 1)
        s = conv2d_kwargs.get("stride", 1)
        padding = (k - 1) * (d + s - 1) / (2 * s)
        assert padding == int(padding), f"padding should be an integer, was {padding:.2f}"
        padding = int(padding)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            **conv2d_kwargs
        )
        # We use LayerNorm here since the size of the input "images" may vary based on the board size
        #self.norm1 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.norm1 = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()
        self.act1 = activation()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            **conv2d_kwargs
        )
        #self.norm2 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.norm2 = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()
        self.final_act = activation()

        if in_channels != out_channels:
            self.change_n_channels = nn.Conv2d(in_channels, out_channels, (1, 1))
        else:
            self.change_n_channels = nn.Identity()

        if squeeze_excitation:
            self.squeeze_excitation = SELayer(out_channels)
        else:
            self.squeeze_excitation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.act1(self.norm1(x))
        x = self.conv2(x)
        x = self.squeeze_excitation(self.norm2(x))
        x = x + self.change_n_channels(identity)
        return self.final_act(x)


# Define the model
class ActionPredictionModel(nn.Module):
    def __init__(self):
        super(ActionPredictionModel, self).__init__()
        # Convolutional encoder
        # Input: (B, 1, 24, 24) or (B, channels, 24, 24)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.res1 = ResidualBlock(64, 256, kernel_size=5, normalize=True, activation=nn.LeakyReLU, squeeze_excitation=True)
        self.res2 = ResidualBlock(256, 256, kernel_size=5, normalize=True, activation=nn.LeakyReLU, squeeze_excitation=True)

        #self.res1 = ResidualBlockOld(64, 128)
        #self.res2 = ResidualBlockOld(128, 128)


        self.res3 = ResidualBlock(512, 512)
        self.res4 = ResidualBlock(512, 512)
        self.res5 = ResidualBlock(512, 512)
        self.res6 = ResidualBlock(512, 512)
        self.res7 = ResidualBlock(512, 512)
        self.res8 = ResidualBlock(512, 512)

        # 64 feature maps of size 24×24 -> gather or flatten + linear
        #self.decoder = nn.Linear(512 + 2 * 0, 17 * 17)
        #self.decoder = nn.Linear(128, 17 * 17)
        self.decoder = nn.Sequential(
            nn.Linear(256, 15*15),
            #nn.ReLU(),
            #nn.Linear(128, 17*17),
        )

        #self.fc = nn.Linear(512, 512)  # Expand channels

# 128 x 8 = 89
# 256 x 8 = 0.996
# 512 = 100% acc


    def forward(self, x, coords):
        #x = x.view(B, -1)
        # Pass through encoder
        x = self.initial(x)

        x = self.res1(x)
        x = self.res2(x)

        x = x.view(B, -1, 24*24)
        indexes = coords
        indexes = indexes.view(-1)

        batch_indexes_my = torch.repeat_interleave(torch.arange(B), 16)

        selected = x[batch_indexes_my, :, indexes]
        valid_features = selected.view(B, 16, -1)

        logits = self.decoder(valid_features)  # Shape: (batch_size, 289, 24, 24)


        #logits = self.fc(valid_features.view(-1, 512))
        #logits = valid_features.view(-1, 512, 1, 1)
        #logits = self.deconv(logits)


        logits = logits.view(B, 16, 15, 15)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample(id):
    my_units = np.zeros((24, 24), dtype=int)
    enemy_units = np.zeros((24, 24), dtype=int)

    # Generate 16 unique random positions for my_units
    my_unit_positions = np.random.choice(24 * 24, size=16, replace=False)
    my_unit_coords = [(pos // 24, pos % 24) for pos in my_unit_positions]

    # Place my units on the grid
    for x, y in my_unit_coords:
        my_units[x, y] = 1

    # Generate 16 unique random positions for enemy_units, avoiding my_units positions
    all_positions = set(range(24 * 24))
    available_positions = all_positions - set(my_unit_positions)
    enemy_unit_positions = np.random.choice(list(available_positions), size=16, replace=False)
    enemy_unit_coords = [(pos // 24, pos % 24) for pos in enemy_unit_positions]

    # Place enemy units on the grid
    for x, y in enemy_unit_coords:
        enemy_units[x, y] = 1

    expected_output = np.zeros((16, 15, 15))

    cords = []
    for p_id, c in enumerate(my_unit_coords):
        x, y = c
        cords.append(x * 24 + y)
        for i in range(-8, 9):
            for j in range(-8, 9):
                xx = x + i
                yy = y + j
                if xx < 0 or yy < 0 or xx >= 24 or yy >= 24:
                    continue
                if enemy_units[xx, yy]:
                    expected_output[p_id, i + 8, j + 8] = True

    return torch.tensor(enemy_units).to(device, dtype=torch.float32), torch.tensor(cords).to(device), torch.tensor(expected_output).to(device, dtype=torch.float32)



import multiprocessing as mp

def create_batch(sample_func, device, batch_size):
    # Function to generate a single sample
    batch_x = []
    batch_c = []
    batch_y = []

    # Use multiprocessing to generate samples in parallel
    #with mp.Pool(processes=min(mp.cpu_count(), batch_size)) as pool:
    #    samples = pool.map(sample_func, range(batch_size))

    samples = []

    for i in range(batch_size):
        samples.append(sample(0))

    for X, C, Y in samples:
        batch_x.append(X.view(1, 1, 24, 24))
        batch_c.append(C.view(1, 16))
        batch_y.append(Y.view(1, 16, 15, 15))

    # Concatenate and move to device
    X = torch.cat(batch_x).to(device)
    C = torch.cat(batch_c).to(device)
    Y = torch.cat(batch_y).to(device)

    return X, C, Y



if __name__ == '__main__':
    # Initialize the model
    model = ActionPredictionModel()

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Suitable for logits
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Mixed precision GradScaler
    scaler = torch.amp.GradScaler('cuda')

    # Example input and target data
    epochs = 10000
    max_acc = 0
    #B = 32  # Batch size

    for epoch in range(epochs):
        if False and epoch == 10:
            run = wandb.init(
                # Set the project where this run will be logged
                project="script",
            )

        X, C, Y = create_batch(sample, device, B)

        model.train()
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            output = model(X, C)  # Output: (batch_size, 16, 17, 17)
            Y = Y.view(B, 16, 15, 15)
            loss = criterion(output, Y)

        # Backward pass with GradScaler
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)  # Unscale gradients before clipping
        params_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients

        # Step optimizer with GradScaler
        scaler.step(optimizer)
        scaler.update()

        # Compute Metrics
        with torch.no_grad():
            probabilities = torch.sigmoid(output)  # Convert logits to probabilities
            predictions = (probabilities > 0.5).float()  # Threshold probabilities

            # Flatten tensors for metric calculation
            Y_flat = Y.view(-1).cpu().numpy()
            predictions_flat = predictions.view(-1).cpu().numpy()

            # Calculate metrics
            precision = precision_score(Y_flat, predictions_flat, zero_division=0)
            recall = recall_score(Y_flat, predictions_flat, zero_division=0)
            accuracy = accuracy_score(Y_flat, predictions_flat)

        max_acc = max(max_acc, accuracy)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.9f}, Precision: {precision:.9f}, Recall: {recall:.9f}, Accuracy: {max_acc:.9f}, Norm: {params_norm:.9f}")

        if False and epoch >= 10:
            wandb.log({'loss': loss.item(), 'precision': precision, 'recall': recall, 'accuracy': accuracy}, step=epoch)

    if False:
        wandb.finish()