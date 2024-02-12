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
        title: str = "Attention Mechanism Heatmap",
        xlable: str = "Keys",
        ylable: str = "Queries",
    ) -> None:
        """
        Plot a heat map chart.

        :param ts: Description.
        :type ts: Tensor
        :return: None
        :rtype: None
        """
        fig, ax = plt.subplots()
        heatmap = ax.imshow(ts, cmap="hot", interpolation="nearest")

        # Add a colorbar to indicate value ranges
        fig.colorbar(heatmap)

        # Set labels
        ax.set_xlabel(xlable)
        ax.set_ylabel(ylable)

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
