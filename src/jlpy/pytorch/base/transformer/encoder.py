"""
Encoder Module for Transformer.

Base implementation for a transformer's encoder.

:Author:  JLDP
:Version: 2024.02.16.01

"""
import torch
import torch.nn as nn


class EBlock(nn.modules):
    """
    Encoder Base Block for a RNN implementation.

    Build a base RNN block for creating a transformer's encoder.

    .. card::
    """

    def __init__(self) -> None:
        """Construct a class instance."""
        super().__init__()
        pass
