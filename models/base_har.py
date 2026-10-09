import torch
import torch.nn as nn

class BaseHARModel(nn.Module):
    """
    Abstract base class providing general utilities for all HAR deep learning models,
    including model checkpointing, size checks, and parameter profiling.
    """
    def __init__(self):
        super().__init__()

    def save_checkpoint(self, path):
        """
        Saves model weights to specified path.
        """
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path, device=torch.device('cpu')):
        """
        Loads model weights from specified path.
        """
        self.load_state_dict(torch.load(path, map_location=device))

    def get_num_parameters(self):
        """
        Returns total and trainable parameter count for model comparison.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
