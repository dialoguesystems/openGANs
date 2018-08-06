import torch
import torch.nn as nn
import torchtext

class MemoryReader(nn.Module):
    def __init__(self):
        super(MemoryReader, self).__init__()

    def _memory_reader(self):
        """
        Remove a checkpoint
        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """
        raise NotImplementedError()