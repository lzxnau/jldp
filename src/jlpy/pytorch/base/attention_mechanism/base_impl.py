"""
Attention Mechanism Module.

Attention mechanism module for testing purposes.

:Author:  JLDP
:Version: 2024.02.12.01

"""
import torch
from torch import Tensor


class AMBase:
    """
    Attention Mechanism Base Class.

    .. card::
    """

    def __init__(self) -> None:
        """Construct a class instance."""
        pass

    def show_heatmap(self, ts: Tensor) -> None:
        """
        Plot a heat map chart.

        :param ts: Description.
        :type ts: Tensor
        :return: None
        :rtype: None
        """
        print(ts)


if __name__ == "__main__":
    amb = AMBase()

    ts = torch.eye(10).reshape((1, 1, 10, 10))
    amb.show_heatmap(ts)
