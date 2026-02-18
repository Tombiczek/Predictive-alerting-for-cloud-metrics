import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    1D CNN for time-series incident prediction.

    Input shape: (batch_size, window_size, 1)
    Output shape: (batch_size, 1)
    """

    def __init__(
        self,
        window_size: int,
        filters: list[int] = [32, 64],
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout_rate: float = 0.3,
        dense_units: int = 64,
    ):
        super().__init__()

        self.window_size = window_size

        # Convolutional layers
        layers = []
        in_channels = 1
        current_length = window_size

        for out_channels in filters:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool_size))
            layers.append(nn.Dropout(dropout_rate))

            in_channels = out_channels
            current_length = current_length // pool_size

        self.conv_layers = nn.Sequential(*layers)

        # Calculate flattened size
        self.flatten_size = filters[-1] * current_length

        # Dense layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch_size, window_size, 1)
        # Conv1d expects: (batch_size, channels, length)
        x = x.transpose(1, 2)  # -> (batch_size, 1, window_size)

        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x


def build_model(config: dict) -> nn.Module:
    """Build a CNN1D model from config."""
    return CNN1D(
        window_size=config["window_size"],
        filters=config.get("filters", [32, 64]),
        kernel_size=config.get("kernel_size", 3),
        pool_size=config.get("pool_size", 2),
        dropout_rate=config.get("dropout_rate", 0.3),
        dense_units=config.get("dense_units", 64),
    )

