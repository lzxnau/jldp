"""
Encoder Module for Transformer.

Base implementation for a transformer's encoder.

:Author:  JLDP
:Version: 2024.02.16.01

"""
from torch import nn


class EBlock(nn.Module):
    """
    Encoder base block for a RNN implementation.

    Build a base RNN block for creating a transformer's encoder.

    .. card::
    """

    def __init__(self, p1: None, p2: None) -> None:
        """
        Construct a class instance.

        :param p1: Parameter 1.
        :type p1: None
        :param p2: Parameter 2.
        :type p2: None
        """
        super().__init__()
        self.p1 = p1
        self.p2 = p2

    def forward(self, p1: None, p2: None) -> None:
        """
        Override module's forward method.

        :param p1: Parameter 1.
        :type p1: None
        :param p2: Parameter 2.
        :type p2: None
        :return: Return
        :rtype: None
        """
        return None
