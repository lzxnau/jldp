"""
Linear Regression Base Implementation.

An example shows all steps for training linear regression model from scratch.

:Author:  JLDP
:Version: 2023.12.02.01

"""
import torch
from torch import Tensor, nn


class LRModel(nn.Module):
    """
    Custom Linear Regression Model as a torch.nn.Module.

    .. card::
    """

    def __init__(
        self, features: int = 10, lr: float = 0.03, sigma: float = 0.01
    ) -> None:
        """Construct a class instance."""
        super().__init__()
        self.loss = nn.MSELoss()
        self.lr = lr
        self.w: Tensor = torch.normal(
            0, sigma, (features, 1), requires_grad=True
        )
        self.b: Tensor = torch.zeros(1, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Matrix x product with weights w plus bisa b."""
        y: Tensor = torch.matmul(x, self.w) + self.b
        return y


if __name__ == "__main__":
    lrm = LRModel()
