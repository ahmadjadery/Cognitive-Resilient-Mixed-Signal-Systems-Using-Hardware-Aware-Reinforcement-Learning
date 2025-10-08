import torch

# This file contains globally shared variables and utility functions
# to avoid circular imports.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
