"""
Linear Regression Base Implementation.

An example shows all steps for training linear regression model from scratch.

:Author:  JLDP
:Version: 2023.12.02.01

"""
import torch
from torch import Tensor, nn


class LRModel(nn.Module):
    """Custom Linear Regression Model as a torch.nn.Module."""

    def __init__(self, features: int = 10, sigma: float = 0.01) -> None:
        """Construct a class instance."""
        super().__init__()
        self.w: Tensor = torch.normal(
            0, sigma, (features, 1), requires_grad=True
        )
        self.b: Tensor = torch.zeros(10, requires_grad=True)
        print(self.b)

    def forward(self, x: Tensor) -> Tensor:
        """Matrix x product with weights w plus bisa b."""
        y: Tensor = torch.matmul(x, self.w) + self.b
        return y

    def loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Calculate the avaerage squared loss between real y and predicted y_hat.

        :param y: Real y from dataset.
        :type y: Tensor
        :param y_hat: Predicted y_hat from model.
        :type y_hat: Tensor
        :return: The average squared loss.
        :rtype: Tensor
        """
        sl: Tensor = (y_hat - y) ** 2 / 2
        asl: Tensor = sl.mean()
        return asl


if __name__ == "__main__":
    lrm = LRModel()
