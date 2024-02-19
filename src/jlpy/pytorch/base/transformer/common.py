"""
Pytorch Transformer Common Classes.

Common classes for building up pytorch transformer module.

:Author:  JLDP
:Version: 2024.02.18.01

"""
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Pytorch Module Positional Encoding implementation.

    Build a positional encoding module.

    .. card::
    """

    def __init__(self, p1: None, p2: None) -> None:
        """
        Construct a pytorch module instance.

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
        Override forward method of the pytorch module.

        :param p1: Parameter 1.
        :type p1: None
        :param p2: Parameter 2.
        :type p2: None
        :return: Return
        :rtype: None
        """
        return None
