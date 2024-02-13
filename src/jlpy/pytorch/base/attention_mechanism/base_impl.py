"""
Attention Mechanism Module.

Attention mechanism module for testing purposes.

:Author:  JLDP
:Version: 2024.02.12.01

"""
import matplotlib.pyplot as plt
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

    def show_heatmap(
        self,
        ts: Tensor,
        **kwargs: str,
    ) -> None:
        """
        Plot a heat map chart.

        :param ts: Description.
        :type ts: Tensor
        :return: None
        :rtype: None
        """
        # init dict
        tstr = "Attention Mechanism Heatmap"
        title: str = _ if (_ := kwargs.get("title")) else tstr
        xlabel: str = _ if (_ := kwargs.get("xlabel")) else "Keys"
        ylabel: str = _ if (_ := kwargs.get("ylabel")) else "Queries"
        cmap: str = _ if (_ := kwargs.get("cmap")) else "Reds"

        fig, ax = plt.subplots()
        heatmap = ax.imshow(ts, cmap=cmap, interpolation="nearest")

        # Add a colorbar to indicate value ranges
        fig.colorbar(heatmap, shrink=0.6)

        # Set labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Set title
        ax.set_title(title)

        plt.show()


if __name__ == "__main__":
    amb = AMBase()

    ts = torch.eye(10)
    print(ts)
    ts1 = ts.reshape((1, 1, 10, 10))
    print(ts1)
    amb.show_heatmap(ts)
